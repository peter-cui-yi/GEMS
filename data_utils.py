#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data utilities for loading and saving JSONL files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

Sample = Dict[str, Any]  # Type alias for sample dictionary


def load_jsonl(path: str) -> List[Sample]:
    """从 jsonl 文件读取所有样本，返回列表。"""
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def save_jsonl(samples: List[Sample], path: str) -> None:
    """把若干样本写回 jsonl 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
