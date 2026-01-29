#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Tuple, Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from data_utils import Sample
from encoders import MultimodalEncoder  # 你原来的类型
# 关键：让类型检查不影响运行
try:
    from encoders import Qwen3VLEmbeddingEncoder
except Exception:
    Qwen3VLEmbeddingEncoder = None


class MultimodalDataset(Dataset):
    def __init__(self, samples: List[Sample], image_root: Optional[str], text_with_rule: bool):
        self.samples = samples
        self.image_root = image_root
        self.text_with_rule = text_with_rule

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_path = s.get("orig_image_path")
        if img_path is None:
            return None

        img_path = str(img_path)
        if self.image_root is not None and (not os.path.isabs(img_path)):
            img_path = os.path.join(self.image_root, img_path)

        desc = (s.get("description", "") or "").strip()
        rule = (s.get("rule", "") or "").strip()
        if self.text_with_rule and rule:
            text = f"{desc} [RULE] {rule}"
        else:
            text = desc
        
        return {"image": img_path, "text": text, "original_sample": s}


def custom_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return {
        "images": [x["image"] for x in batch],
        "texts": [x["text"] for x in batch],
        "original_samples": [x["original_sample"] for x in batch],
    }


def _resolve_autocast_dtype(amp_dtype: str) -> Optional[torch.dtype]:
    amp_dtype = (amp_dtype or "none").lower()
    if amp_dtype in ("none", "no", "false", "off"):
        return None
    if amp_dtype in ("fp16", "float16"):
        return torch.float16
    if amp_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown amp_dtype={amp_dtype}, choose from: none|fp16|bf16")


def build_fused_embeddings(
    samples: List[Sample],
    image_root: Optional[str],
    encoder,
    text_with_rule: bool = True,
    batch_size: int = 16,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    amp_dtype: str = "fp16",
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[Sample]]:
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dataset = MultimodalDataset(samples, image_root, text_with_rule)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False,
    )

    valid_samples: List[Sample] = []
    embs_cpu: List[torch.Tensor] = []

    ac_dtype = _resolve_autocast_dtype(amp_dtype)

    iterator = dataloader
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(dataloader, desc="Building embeddings")
        except Exception:
            iterator = dataloader

    # ✅ 核心：如果 encoder 有 encode_fused，就一次 forward 直接出 fused embedding
    use_fused = hasattr(encoder, "encode_fused")

    for batch in iterator:
        if batch is None:
            continue

        images = batch["images"]
        texts = batch["texts"]
        batch_samples = batch["original_samples"]

        try:
            if use_fused:
                # Qwen3-VL-Embedding 推荐走这里：一次 process => [B, D]
                # 注意：大多数 embedder 内部会自己处理 autocast/dtype，
                # 这里保留 inference_mode 以防外层产生 autograd 开销
                with torch.inference_mode():
                    fused = encoder.encode_fused(images=images, texts=texts)  # torch.Tensor [B,D]
            else:
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=ac_dtype):
                    img_emb = encoder.encode_image(images)
                    txt_emb = encoder.encode_text(texts)
                    fused = torch.cat([img_emb, txt_emb], dim=-1)

            embs_cpu.append(fused.detach().to("cpu", dtype=torch.float32))
            valid_samples.extend(batch_samples)

        except Exception:
            continue

    if len(embs_cpu) == 0:
        raise RuntimeError("No valid samples produced embeddings.")

    fused_all = torch.cat(embs_cpu, dim=0).numpy()
    return fused_all, valid_samples
