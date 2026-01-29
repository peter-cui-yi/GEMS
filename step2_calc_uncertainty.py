import os
# --- 显存优化 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import CrossEntropyLoss
from PIL import Image
from transformers import AutoModel, AutoConfig, AutoProcessor
from tqdm import tqdm
import warnings
import glob

warnings.filterwarnings("ignore")

# --- 模型检查 ---
QWEN25VL_AVAILABLE = False
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN25VL_AVAILABLE = True
except ImportError: pass

# --- Dataset 定义 (保持不变) ---
MAX_IMAGE_DIM = 768

class UncertaintyDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    sample = json.loads(line.strip())
                    self.samples.append((idx, sample))
                except json.JSONDecodeError:
                    continue
    
    def __len__(self):
        return len(self.samples)

    def _resize_image(self, image):
        width, height = image.size
        if width <= MAX_IMAGE_DIM and height <= MAX_IMAGE_DIM:
            return image
        if width > height:
            new_width = MAX_IMAGE_DIM
            new_height = int(height * (MAX_IMAGE_DIM / width))
        else:
            new_height = MAX_IMAGE_DIM
            new_width = int(width * (MAX_IMAGE_DIM / height))
        return image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    def __getitem__(self, idx):
        
        #sample_idx, sample_data = self.samples[idx]
        
        # 修正：Subset 传入的 idx 是原数据集的绝对索引，所以直接取
        real_idx, sample_data = self.samples[idx]
        
        image_path = sample_data.get('orig_image_path')
        image = None
        if image_path and os.path.exists(image_path):
            try:
                raw_image = Image.open(image_path).convert('RGB')
                image = self._resize_image(raw_image)
            except: pass
            
        # 构建对话
        hazard_desc = sample_data.get('hazard_desc', 'No description')
        hazard_category = sample_data.get('hazard_category', 'Unknown')
        regulation = sample_data.get('regulation', '')
        
        user_content = "Analyze this construction site image and identify any safety hazards or violations."
        assistant_content = f"Hazard Category: {hazard_category}\nDescription: {hazard_desc}"
        if regulation: assistant_content += f"\nRegulation: {regulation}"

        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_content}]},
            {"role": "assistant", "content": assistant_content}
        ]
        return real_idx, image, conversation

def setup_model(model_path, device_id):
    device = f"cuda:{device_id}"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 自动选择模型类
    if getattr(config, 'model_type', None) == "qwen2_5_vl" and QWEN25VL_AVAILABLE:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        # Fallback
        from transformers import Qwen2VLForConditionalGeneration
        model_class = Qwen2VLForConditionalGeneration

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None # 手动控制设备
    ).to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor, device

# --- Collator ---
def run_collate_fn(batch, processor):
    indices, images, convs, has_imgs = [], [], [], []
    dummy = Image.new('RGB', (28, 28), color='black')
    for idx, img, conv in batch:
        indices.append(idx)
        if img:
            images.append(img)
            has_imgs.append(True)
        else:
            images.append(dummy)
            has_imgs.append(False)
        convs.append(conv)
    
    texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convs]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    
    # Label Masking
    labels = inputs["input_ids"].clone()
    try:
        im_start = processor.tokenizer.encode("<|im_start|>")[0]
        assistant = processor.tokenizer.encode("assistant")[0]
        for i in range(labels.shape[0]):
            ids = inputs["input_ids"][i].tolist()
            mask_idx = 0
            # 寻找最后一个 assistant 头
            for j in range(len(ids) - 2):
                if ids[j] == im_start and ids[j+1] == assistant:
                    mask_idx = j + 2
            labels[i, :mask_idx] = -100
    except: pass
    
    inputs["labels"] = labels
    return {"indices": indices, "inputs": inputs, "has_image": has_imgs}

class DataCollator:
    def __init__(self, processor): self.processor = processor
    def __call__(self, batch): return run_collate_fn(batch, self.processor)

def compute_nll(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.shape)
    return (loss.sum(dim=1) / (shift_labels != -100).sum(dim=1).float().clamp(min=1.0)).detach().cpu().numpy()

# --- Worker Function ---
def worker_inference(rank, gpu_id, assigned_indices, args):
    """
    rank: 进程号 (0..7)
    gpu_id: 实际使用的 GPU ID
    assigned_indices: 主进程分配好的绝对索引列表
    """
    try:
        # Load Dataset (Lightweight, mainly metadata)
        full_dataset = UncertaintyDataset(args.input_jsonl)
        
        # Create Subset using EXPLICIT indices
        subset = Subset(full_dataset, assigned_indices)
        
        print(f"[GPU {gpu_id}] Loaded Subset: {len(subset)} samples.")
        
        model, processor, device = setup_model(args.model_path, gpu_id)
        
        dataloader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=DataCollator(processor),
            num_workers=0,
            pin_memory=True
        )
        
        results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"GPU {gpu_id}", position=rank):
                batch_indices = batch["indices"]
                has_image = batch["has_image"]
                inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
                
                try:
                    outputs = model(**inputs)
                    scores = compute_nll(outputs.logits, inputs["labels"])
                    for i, score in enumerate(scores):
                        results.append((batch_indices[i], float(score) if has_image[i] else 0.5))
                except Exception as e:
                    print(f"Error batch {batch_indices[0]}: {e}")
                    for idx in batch_indices: results.append((idx, 0.5))
                    torch.cuda.empty_cache()
        
        # Save explicit part file
        results_array = np.array(results, dtype=np.float32)
        temp_file = f"{args.output_npy}.part{rank}" # Use RANK not GPU_ID for file naming
        np.save(temp_file, results_array)
        print(f"[GPU {gpu_id}] Saved {len(results)} scores to part{rank}")
        
    except Exception as e:
        print(f"Worker {rank} Failed: {e}")
        import traceback; traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_npy", type=str, default="../features/scores.npy")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    args = parser.parse_args()

    # 1. Cleanup Old Files
    print("🧹 Cleaning up old part files...")
    for f in glob.glob(f"{args.output_npy}.part*"):
        os.remove(f)

    # 2. Prepare Indices (Foolproof Splitting)
    print("📦 Preparing Indices...")
    full_dataset = UncertaintyDataset(args.input_jsonl)
    total_samples = len(full_dataset)
    print(f"   Total Samples: {total_samples}")
    
    all_indices = np.arange(total_samples)
    # 核心修复：直接切分索引数组
    chunks = np.array_split(all_indices, args.num_gpus)
    
    print(f"   Split into {len(chunks)} chunks.")
    print(f"   Chunk sizes: {[len(c) for c in chunks]}")

    # 3. Start Processes
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(args.num_gpus):
        indices = chunks[rank]
        # map rank to gpu_id (usually 1:1)
        p = mp.Process(target=worker_inference, args=(rank, rank, indices, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    # 4. Merge Results
    print("🔄 Merging Results...")
    final_results_list = []
    
    # 按 rank 顺序读取，保证有序
    for rank in range(args.num_gpus):
        part_file = f"{args.output_npy}.part{rank}.npy" # np.save adds .npy
        if not os.path.exists(part_file):
            print(f"❌ Missing part file: {part_file}")
            continue
            
        data = np.load(part_file)
        final_results_list.append(data)
        
    if not final_results_list:
        print("❌ No data generated.")
        return

    merged = np.concatenate(final_results_list, axis=0)
    
    # Robust Sorting: Sort by original Index to ensure alignment
    # merged is [[index, score], [index, score]...]
    print("   Sorting by index...")
    sorted_indices = np.argsort(merged[:, 0])
    sorted_data = merged[sorted_indices]
    
    # Check Integrity
    final_scores = sorted_data[:, 1]
    
    if len(final_scores) != total_samples:
        print(f"❌ Mismatch! Expected {total_samples}, got {len(final_scores)}")
        # Check for duplicates
        unique_indices = np.unique(sorted_data[:, 0])
        print(f"   Unique Indices found: {len(unique_indices)}")
    else:
        print(f"✅ Count matches: {len(final_scores)}")
        os.makedirs(os.path.dirname(args.output_npy), exist_ok=True)
        np.save(args.output_npy, final_scores)
        print(f"💾 Saved to {args.output_npy}")
        
        # Cleanup
        for rank in range(args.num_gpus):
            f = f"{args.output_npy}.part{rank}.npy"
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()