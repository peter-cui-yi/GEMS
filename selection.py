#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Selection algorithms for data subset selection.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def greedy_selection_with_quota(
    samples: List[Dict[str, Any]],
    value_scores: np.ndarray,
    budget_k: int,
    min_per_class: int = 0,
    max_per_class: Optional[int] = None,
    class_key: str = "category",
) -> List[int]:
    """
    基于综合 value_scores 进行简化贪心选样，并考虑类别配额。

    策略：
    1. 全局按 value_scores 从大到小排序；
    2. 依次尝试选入，若某类未超过 max_per_class 则选入；
    3. 直到达到 budget_k 或遍历结束。
    """
    n = len(samples)
    assert value_scores.shape[0] == n

    indices_sorted = np.argsort(-value_scores)  # 降序
    class_counts: Dict[str, int] = {}
    selected_indices: List[int] = []

    for idx in indices_sorted:
        if len(selected_indices) >= budget_k:
            break

        s = samples[idx]
        cls = str(s.get(class_key, "UNKNOWN"))

        cur_count = class_counts.get(cls, 0)
        if max_per_class is not None and cur_count >= max_per_class:
            continue

        selected_indices.append(idx)
        class_counts[cls] = cur_count + 1

    print(f"Selected {len(selected_indices)} / {budget_k} samples.")
    print("Class distribution in selected set:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")

    return selected_indices
