#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph building and scoring utilities for UniMax-style data selection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
from tqdm import tqdm


class UnimaxGraphBuilder(nn.Module):
    """
    UniMax 风格的 Graph Builder：
    - 输入：节点特征 X（由多模态 encoder 得到），形状 [N, D]
    - 输出：Q', K'（做完 ReLU + L2 Norm），用于构建注意力相似度矩阵。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算：
        - X: [N, D]，float32
        返回：
        - Qp, Kp: 经过 ReLU + L2 Norm 的 Q', K'，形状 [N, H]
        """
        Q = self.W_q(X)  # [N, H]
        K = self.W_k(X)  # [N, H]

        # ReLU 激活（非负），然后按行做 L2 归一化
        Q = F.relu(Q)
        K = F.relu(K)

        Q = F.normalize(Q, p=2, dim=-1)  # [N, H]
        K = F.normalize(K, p=2, dim=-1)  # [N, H]

        return Q, K


@torch.no_grad()
def build_graph_unimax_topk(
    embeddings: np.ndarray,
    k: int = 32,
    hidden_dim: int = 256,
    device: str = "cpu",
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 UniMax 风格的公式构建稀疏 kNN 图：
      A ≈ f_sp(Q'K'^T) + f_sp(X_norm X_norm^T)
    然后对每一行做 Top-k，得到：
      - knn_indices: [N, k]
      - knn_similarities: [N, k]

    参数：
    - embeddings: [N, D] 的 numpy 数组，融合后的多模态特征 X
    - k: 每个节点保留的邻居数
    - hidden_dim: Q/K 的隐藏维度
    - device: "cpu" 或 "cuda"
    - chunk_size: 分块计算的行数，避免一次性 N×N 占用太大内存

    注意：
    - 这里为了清晰，没有做 KMeans 分桶；大规模（> 2~3 万）时建议你加一层 cluster。
    - 目前 GraphBuilder 未训练，只是随机初始化，主要还是靠 X_norm X_norm^T（余弦相似度），
      Q'K'^T 起一个 re-weight / perturb 的作用。
    """
    N, D = embeddings.shape
    if k >= N:
        k = N - 1

    # numpy -> torch
    X = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)  # [N, D]

    # 1) 归一化 X，用于余弦相似度
    X_norm = F.normalize(X, p=2, dim=-1)  # [N, D]

    # 2) 构建 UniMax 风格的 Q', K'
    graph_builder = UnimaxGraphBuilder(input_dim=D, hidden_dim=hidden_dim).to(device)
    graph_builder.eval()
    Qp, Kp = graph_builder(X)  # [N, H], [N, H]

    # 3) 分块计算每一行的 Top-k
    # 结果容器（numpy），方便后续处理
    knn_indices = np.zeros((N, k), dtype=np.int32)
    knn_similarities = np.zeros((N, k), dtype=np.float32)

    # 将 Kp 和 X_norm 固定为 torch Tensor（避免重复转）
    Kp_T = Kp.t()      # [H, N]
    Xnorm_T = X_norm.t()  # [D, N]

    # 逐块处理行 i：每块大小为 chunk_size
    for start in tqdm(range(0, N, chunk_size), desc="Building UniMax graph (Top-k)"):
        end = min(start + chunk_size, N)
        B = end - start

        # 当前块的 Q' 和 X_norm （行）
        Q_chunk = Qp[start:end, :]      # [B, H]
        Xn_chunk = X_norm[start:end, :]  # [B, D]

        # 注意力相似度部分：S_att = Q_chunk @ Kp_T  => [B, N]
        S_att = torch.matmul(Q_chunk, Kp_T)  # [B, N]

        # 余弦相似度部分：S_sim = Xn_chunk @ Xnorm_T => [B, N]
        S_sim = torch.matmul(Xn_chunk, Xnorm_T)  # [B, N]

        # 合并：这里简单相加，你也可以加权
        S_total = S_att + S_sim  # [B, N]

        # 自环（自己和自己）对应位置设为 -inf，避免选到自己
        row_indices = torch.arange(start, end, device=device)  # [B]
        S_total[torch.arange(B, device=device), row_indices] = -1e9

        # 取每行 Top-k
        # topk_values: [B, k], topk_idx: [B, k]（全局索引）
        topk_values, topk_idx = torch.topk(S_total, k=k, dim=-1)  # 最大的 k 个相似度

        # 写入 numpy 结果
        knn_indices[start:end, :] = topk_idx.cpu().numpy().astype(np.int32)
        knn_similarities[start:end, :] = topk_values.cpu().numpy().astype(np.float32)

    return knn_indices, knn_similarities


def compute_influence_score(
    knn_indices: np.ndarray,
    knn_similarities: np.ndarray,
) -> np.ndarray:
    """
    计算每个样本的"影响力分数"（influence）。

    简化定义：
    - influence(i) = 与所有 k 个近邻的相似度之和
      = sum_j sim(i, j)
    """
    influence = knn_similarities.sum(axis=1)  # [N]
    return influence.astype("float32")


def compute_language_uncertainty(
    samples: List[Dict[str, Any]],
) -> np.ndarray:
    """
    语言不确定性（占位实现版本）。

    正式版本中：
    - 用 teacher 模型对每个样本的参考答案/描述计算平均 token NLL 等。

    这里：
    - 如果 sample 中已有 'uncertainty' 字段，则直接使用；
    - 否则全部设为 1.0（等价于只看 influence）。
    """
    vals = []
    for s in samples:
        if "uncertainty" in s:
            vals.append(float(s["uncertainty"]))
        else:
            vals.append(1.0)
    return np.array(vals, dtype="float32")


def min_max_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """简单的 min-max 归一化到 [0, 1] 区间。"""
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < eps:
        return np.full_like(x, 0.5)
    return (x - x_min) / (x_max - x_min)
