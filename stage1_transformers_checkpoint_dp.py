#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 (checkpoint + chunk DP): Pre-compute fused multimodal embeddings.

- --chunk_offset / --chunk_stride：多进程/多卡分配 chunk
- --merge_only：只合并，不做编码
- 优化 GPU util：
  * DataLoader: persistent_workers + prefetch_factor + pin_memory
  * inference_mode + autocast(fp16/bf16) + TF32
  * torch.cat on GPU, move to CPU once per batch
"""

import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch

from data_utils import load_jsonl, save_jsonl, Sample
from encoders import DummyMultimodalEncoder, Qwen25VLEncoder, MultimodalEncoder, Qwen3VLEmbeddingEncoder
from build_fused import build_fused_embeddings


def get_chunk_paths(output_npy: str, output_jsonl: str, chunk_idx: int) -> Tuple[str, str]:
    npy_prefix, _ = os.path.splitext(output_npy)
    jsonl_prefix, _ = os.path.splitext(output_jsonl)
    emb_chunk = f"{npy_prefix}.chunk{chunk_idx:05d}.npy"
    jsonl_chunk = f"{jsonl_prefix}.chunk{chunk_idx:05d}.jsonl"
    return emb_chunk, jsonl_chunk


def process_one_chunk(
    all_samples: List[Sample],
    image_root: Optional[str],
    encoder: MultimodalEncoder,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    amp_dtype: str,
    chunk_idx: int,
    chunk_start: int,
    chunk_end: int,
    emb_chunk_path: str,
    jsonl_chunk_path: str,
) -> Tuple[int, int]:
    chunk_samples = all_samples[chunk_start:chunk_end]
    num_input = len(chunk_samples)
    print(f"[Chunk {chunk_idx}] Processing samples[{chunk_start}:{chunk_end}] (total {num_input}) ...")

    fused_embs, valid_samples = build_fused_embeddings(
        samples=chunk_samples,
        image_root=image_root,
        encoder=encoder,
        text_with_rule=True,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        amp_dtype=amp_dtype,
        show_progress=True,
    )

    np.save(emb_chunk_path, fused_embs.astype("float32"))
    save_jsonl(valid_samples, jsonl_chunk_path)

    print(
        f"[Chunk {chunk_idx}] Done. Valid={len(valid_samples)}, embeddings shape={fused_embs.shape}. "
        f"Saved to:\n  {emb_chunk_path}\n  {jsonl_chunk_path}"
    )
    return num_input, len(valid_samples)


def merge_all_chunks(output_npy: str, output_jsonl: str, num_chunks: int) -> None:
    print("\nMerging all chunks into final outputs...")

    npy_prefix, _ = os.path.splitext(output_npy)
    jsonl_prefix, _ = os.path.splitext(output_jsonl)

    all_embs: List[np.ndarray] = []
    all_samples: List[Sample] = []
    total_valid = 0

    for chunk_idx in range(num_chunks):
        emb_chunk_path = f"{npy_prefix}.chunk{chunk_idx:05d}.npy"
        jsonl_chunk_path = f"{jsonl_prefix}.chunk{chunk_idx:05d}.jsonl"

        if not (os.path.exists(emb_chunk_path) and os.path.exists(jsonl_chunk_path)):
            print(f"[Merge] Warning: chunk {chunk_idx} missing, skipping.")
            continue

        emb_chunk = np.load(emb_chunk_path)
        samples_chunk: List[Sample] = load_jsonl(jsonl_chunk_path)

        if emb_chunk.shape[0] != len(samples_chunk):
            raise ValueError(
                f"[Merge] Mismatch in chunk {chunk_idx}: embeddings={emb_chunk.shape[0]}, samples={len(samples_chunk)}."
            )

        all_embs.append(emb_chunk)
        all_samples.extend(samples_chunk)
        total_valid += emb_chunk.shape[0]
        print(f"[Merge] Loaded chunk {chunk_idx}: {emb_chunk.shape[0]} samples.")

    if not all_embs:
        raise RuntimeError("No chunk files found to merge.")

    fused_embs_all = np.concatenate(all_embs, axis=0)
    print(f"[Merge] Concatenated embeddings: shape={fused_embs_all.shape}, total_valid={total_valid}")

    np.save(output_npy, fused_embs_all.astype("float32"))
    save_jsonl(all_samples, output_jsonl)

    print(f"[Merge] Saved final embeddings to {output_npy}\n[Merge] Saved final valid samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 (checkpoint + chunk DP) using Transformers encoder.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_npy", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)

    parser.add_argument("--encoder_type", type=str, default="qwen2vl", choices=["dummy", "qwen2vl","qwen3vl_embed"])
    parser.add_argument("--qwen_model_path", type=str, required=True)
    parser.add_argument("--encoder_device", type=str, default="cuda", help="建议每进程用 cuda，并用 CUDA_VISIBLE_DEVICES 绑卡")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size_samples", type=int, default=2000)

    # 多进程/多卡并行分配 chunk
    parser.add_argument("--chunk_offset", type=int, default=0)
    parser.add_argument("--chunk_stride", type=int, default=1)

    # 只合并
    parser.add_argument("--merge_only", action="store_true")

    # ✅ DataLoader / AMP 参数（新增）
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["none", "fp16", "bf16"])

    args = parser.parse_args()

    print(f"Loading data from {args.input_jsonl} ...")
    all_samples: List[Sample] = load_jsonl(args.input_jsonl)
    total_samples = len(all_samples)
    print(f"Loaded {total_samples} samples.")

    chunk_size = int(args.chunk_size_samples)
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    print(f"Will process in {num_chunks} chunks, each with up to {chunk_size} raw samples.")

    if args.merge_only:
        merge_all_chunks(args.output_npy, args.output_jsonl, num_chunks=num_chunks)
        print("Merge finished.")
        return

    # 初始化编码器：每进程一份模型，绑定到本进程可见的那张卡
    if args.encoder_type == "dummy":
        encoder = DummyMultimodalEncoder(image_dim=512, text_dim=512, seed=42)
    elif args.encoder_type == "qwen2vl":
        encoder = Qwen25VLEncoder(model_path=args.qwen_model_path, device=args.encoder_device)
    elif args.encoder_type == "qwen3vl_embed":
       # ✅ Qwen3-VL-Embedding
        torch_dtype = torch.float16 if args.amp_dtype == "fp16" else (torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32)
        encoder = Qwen3VLEmbeddingEncoder(
            model_path=args.qwen_model_path,
            instruction="Represent construction-site safety hazards and regulations for retrieval and clustering.",
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            max_length=8192,
            normalize=True,
        )
    else:
        raise ValueError(f"Invalid encoder type: {args.encoder_type}")

    for chunk_idx in range(num_chunks):
        if (chunk_idx - int(args.chunk_offset)) % int(args.chunk_stride) != 0:
            continue

        chunk_start = chunk_idx * chunk_size
        chunk_end = min((chunk_idx + 1) * chunk_size, total_samples)

        emb_chunk_path, jsonl_chunk_path = get_chunk_paths(args.output_npy, args.output_jsonl, chunk_idx)

        if os.path.exists(emb_chunk_path) and os.path.exists(jsonl_chunk_path):
            print(f"[Chunk {chunk_idx}] Found existing chunk files, skip.")
            continue

        process_one_chunk(
            all_samples=all_samples,
            image_root=args.image_root,
            encoder=encoder,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            amp_dtype=args.amp_dtype,
            chunk_idx=chunk_idx,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            emb_chunk_path=emb_chunk_path,
            jsonl_chunk_path=jsonl_chunk_path,
        )

    print("This worker finished its assigned chunks.")


if __name__ == "__main__":
    main()
