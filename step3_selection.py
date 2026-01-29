import os
import argparse
import json
import numpy as np
import scipy.sparse as sp
import faiss
import heapq
from tqdm import tqdm
import time
import sys

# --- 默认路径配置 ---
DEFAULT_EMB = "/hpc2hdd/home/ycui785/UnimaxMM/main_algo_code/features/llava_final_embeddings.npy"     # Step 1 输出
DEFAULT_SCORE = "/hpc2hdd/home/ycui785/UnimaxMM/main_algo_code/features/llava_final_embeddings_scores.npy"    # Step 2 输出
DEFAULT_JSONL = "/hpc2hdd/home/ycui785/UnimaxMM/main_algo_code/features/llava_final_valid.jsonl"      # Step 1 输出的元数据
DEFAULT_OUTPUT = "unimax_selected_llava_1.json"

class UniMaxSelector:
    def __init__(self, emb_path, score_path, jsonl_path, k_neighbors=20):
        print(f"Loading resources...")
        
        # 1. 加载 Embeddings (使用 mmap 节省内存)
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        self.X = np.load(emb_path, mmap_mode='r')
        
        # 2. 加载 Scores
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Scores not found: {score_path}")
        self.scores = np.load(score_path)
        
        # 3. 加载元数据 (用于最后导出)
        self.jsonl_path = jsonl_path
        
        # 4. 校验对齐
        self.N = self.X.shape[0]
        if self.scores.shape[0] != self.N:
            raise ValueError(f"Mismatch! Emb: {self.N}, Scores: {self.scores.shape[0]}")
        
        # 处理可能的 NaN
        if np.isnan(self.scores).any():
            print("⚠️ Warning: NaNs found in scores. Replacing with 0.5.")
            self.scores = np.nan_to_num(self.scores, nan=0.5)

        self.K = k_neighbors
        self.Graph = None
        
        print(f"✅ Data Loaded. N={self.N}, Dim={self.X.shape[1]}")

    def build_graph(self):
        """
        构建 k-NN 稀疏图 (使用 FAISS GPU 加速)
        """
        print(f"Building k-NN Graph (k={self.K})...")
        t0 = time.time()
        
        # 准备数据 (FAISS 需要 float32)
        # 注意：这里必须把 mmap 数据读入内存才能转 float32，如果内存不够，需要分块处理
        # 84k 数据占用内存很小，直接读没问题
        X_float = self.X.astype(np.float32)
        faiss.normalize_L2(X_float) # Cosine Similarity 前置步骤
        
        d = X_float.shape[1]
        
        # 使用 GPU 资源
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(d) # Inner Product = Cosine after norm
        
        try:
            # 尝试使用 GPU
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
            gpu_index.add(X_float)
            print("   -> FAISS Index built on GPU.")
            
            # 搜索 Top-K
            D, I = gpu_index.search(X_float, self.K)
        except Exception as e:
            print(f"⚠️ GPU failed ({e}), falling back to CPU...")
            index_flat.add(X_float)
            D, I = index_flat.search(X_float, self.K)

        # 构建稀疏矩阵 (CSR)
        # D 是相似度 (Cosine)，I 是索引
        row = np.repeat(np.arange(self.N), self.K)
        col = I.flatten()
        data = D.flatten()
        
        # 过滤负值 (Cosine 极少数情况)
        data = np.maximum(data, 0)
        
        self.Graph = sp.coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()
        print(f"✅ Graph constructed in {time.time()-t0:.2f}s. Edges: {self.Graph.nnz}")

    def influence_gain(self, node_idx, current_influence, saturation_threshold):
        """
        计算增益 (Submodular Marginal Gain)
        """
        # 获取邻居和权重
        start_ptr = self.Graph.indptr[node_idx]
        end_ptr = self.Graph.indptr[node_idx + 1]
        neighbors = self.Graph.indices[start_ptr:end_ptr]
        weights = self.Graph.data[start_ptr:end_ptr]
        
        # 传播值 = 边权重 * 节点自身的 Uncertianty
        # 逻辑：选了这个点，它的所有邻居都会收到一份“知识”
        propagation = weights * self.scores[node_idx]
        
        # 计算增益：min(T, old + prop) - min(T, old)
        old_vals = current_influence[neighbors]
        new_vals = old_vals + propagation
        
        gains = np.minimum(saturation_threshold, new_vals) - np.minimum(saturation_threshold, old_vals)
        return np.sum(gains)

    def select(self, budget_percent=0.05, saturation_threshold=0.8):
        """
        CELF (Lazy Greedy) 算法核心
        """
        budget = int(self.N * budget_percent)
        print(f"Starting Selection (Budget={budget}, Threshold={saturation_threshold})...")
        
        selected_set = []
        current_influence = np.zeros(self.N, dtype=np.float32)
        pq = [] # Priority Queue (min-heap, store negative gain)
        
        # 1. 初始化堆 (Initialization)
        print("   -> Initializing Priority Queue...")
        # 这一步比较慢，因为要算 N 次增益，可以优化，但几万条数据还能接受
        for i in tqdm(range(self.N), desc="Init PQ"):
            gain = self.influence_gain(i, current_influence, saturation_threshold)
            heapq.heappush(pq, (-gain, i))
            
        # 2. 选择循环 (Selection Loop)
        print("   -> Running Lazy Greedy...")
        t0 = time.time()
        
        with tqdm(total=budget) as pbar:
            while len(selected_set) < budget and pq:
                neg_gain, node_idx = heapq.heappop(pq)
                
                # Lazy Check (CELF Magic)
                # 重新计算该节点的真实增益
                real_gain = self.influence_gain(node_idx, current_influence, saturation_threshold)
                
                # 查看堆顶（目前第二好）的旧增益
                # 如果当前节点的真实增益，依然比第二名的旧增益大，那它肯定是冠军
                if not pq or real_gain >= -pq[0][0]:
                    selected_set.append(node_idx)
                    
                    # 更新全局状态
                    start_ptr = self.Graph.indptr[node_idx]
                    end_ptr = self.Graph.indptr[node_idx + 1]
                    neighbors = self.Graph.indices[start_ptr:end_ptr]
                    weights = self.Graph.data[start_ptr:end_ptr]
                    
                    propagation = weights * self.scores[node_idx]
                    current_influence[neighbors] += propagation
                    
                    pbar.update(1)
                else:
                    # 增益不够大，放回堆里重新排队
                    heapq.heappush(pq, (-real_gain, node_idx))
        
        print(f"✅ Selection finished in {time.time()-t0:.2f}s.")
        return selected_set

    def save_results(self, selected_indices, output_path):
        print(f"Saving {len(selected_indices)} samples to {output_path}...")
        
        # 1. 读取原始 JSONL (只读取选中的行)
        # 为了速度，我们把 index 变成 set
        idx_set = set(selected_indices)
        selected_data = []
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in idx_set:
                    try:
                        selected_data.append(json.loads(line.strip()))
                    except: pass
        
        # 2. 保存为 JSON (LLaVA 训练格式)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)
            
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_path", type=str, default=DEFAULT_EMB)
    parser.add_argument("--score_path", type=str, default=DEFAULT_SCORE)
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--ratio", type=float, default=0.01, help="Selection ratio (e.g., 0.05 for 5%)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Information saturation threshold")
    
    args = parser.parse_args()
    
    selector = UniMaxSelector(args.emb_path, args.score_path, args.jsonl_path)
    
    # 1. 建图
    selector.build_graph()
    
    # 2. 筛选
    indices = selector.select(budget_percent=args.ratio, saturation_threshold=args.threshold)
    
    # 3. 保存
    selector.save_results(indices, args.output_path)