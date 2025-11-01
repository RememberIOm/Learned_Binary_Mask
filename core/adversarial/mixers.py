# core/adversarial/mixers.py

from __future__ import annotations
from typing import Iterable, Dict, Any, Optional
import math
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler

from core.adversarial.samplers import MixedBatchSampler


class _ConcatDataset(torch.utils.data.Dataset):
    """Concatenate two datasets. Robust to tagged indices like ('adv', i) or ('base', i).
    Some custom samplers yield (tag, idx) tuples to preserve source ratios; handle both.
    """

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        la = len(self.a)
        # Accept tagged indices from custom samplers
        if isinstance(idx, tuple) and len(idx) == 2:
            tag, i = idx
            i = int(i)
            if tag == "base":
                return self.a[i]
            if tag == "adv":
                return self.b[i]
            # Unknown tag: fall back to flat addressing
            idx = i
        # Flat (int) addressing
        idx = int(idx)
        return self.a[idx] if idx < la else self.b[idx - la]


def _merge_examples(examples: Any):
    """Default collate: stack tensors and keep dict keys."""
    if isinstance(examples[0], dict):
        out = {}
        for k in examples[0].keys():
            vals = [ex[k] for ex in examples]
            out[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
        return out
    raise ValueError("Expect dict-examples for collate.")


def make_mixed_loader(
    base_ds: Dataset,
    adv_ds: Dataset,
    batch_size: int,
    adv_ratio: float,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn=_merge_examples,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader mixing base and adversarial datasets at adv_ratio.

    Args:
        base_ds: Original dataset (DictSampleDataset or compatible)
        adv_ds: Adversarial dataset
        adv_ratio: Portion of adversarial samples per batch (0..1)
    Notes:
        - This function does NOT generate adversarial samples; use generators.py first.
        - If dataset lengths are highly imbalanced, the last few batches may deviate from the ratio.
    """
    sampler = MixedBatchSampler(
        base_len=len(base_ds),
        adv_len=len(adv_ds),
        batch_size=batch_size,
        adv_ratio=adv_ratio,
        drop_last=drop_last,
        shuffle=shuffle,
    )

    # Materialize an indexable “view” over concatenated datasets for the sampler tuples
    concat = _ConcatDataset(base_ds, adv_ds)

    def _collate(batch):
        """
        Robust collate that accepts either:
          - a list of tagged indices: List[("adv"|"base", int)], or
          - a list of already-materialized dict samples (from ConcatDataset.__getitem__).
        We never assume the shape a priori to avoid mismatches between sampler and dataset.
        """
        # Case 1: tagged indices produced by BatchSampler
        if batch and isinstance(batch[0], tuple) and len(batch[0]) == 2:
            batch_items = []
            for tag, local_idx in batch:
                i = int(local_idx)
                if tag == "base":
                    batch_items.append(base_ds[i])
                else:
                    batch_items.append(adv_ds[i])
            return collate_fn(batch_items)
        # Case 2: already fetched dict samples
        return collate_fn(batch)

    return DataLoader(
        concat,  # unused in __getitem__ path, but required by DataLoader
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )
