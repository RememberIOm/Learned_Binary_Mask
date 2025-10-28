# pipelines/decompose.py
from __future__ import annotations
import torch
from decomposition import decompose_to_target_vs_rest


def decompose_binary(model, target_idx: int, device: torch.device):
    """Return a copy of model whose final head outputs [target, negative]."""
    return decompose_to_target_vs_rest(model, target_idx).to(device)
