# wanda_pruning.py
# --------------------------------------------------------------
# Wanda / N:M Wanda pruning utilities for Linear layers
# - Collect per-input activation norms from a calibration set
# - Score weights as |w_ij| * ||X_j||_2 and prune smallest entries
# - Supports per-output unstructured pruning and N:M structured pruning
# --------------------------------------------------------------

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _layers_to_prune(model: nn.Module) -> Dict[str, nn.Linear]:
    """
    Collect Linear layers except the final classifier (common names: 'classifier', 'classification_head', 'fc').
    """
    final_keys = {"classifier", "classification_head", "fc"}
    skip_names = set()
    for k in final_keys:
        if hasattr(model, k):
            skip_names.add(k)

    targets: Dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            parts = set(name.split("."))
            if skip_names & parts:
                continue
            targets[name] = mod
    return targets


@torch.no_grad()
def _collect_act_norms(
    model: nn.Module, layer_name: str, calib_dl: DataLoader, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Run forward passes and collect L2 norms of the input activations to the given layer.
    Returns a tensor of shape (C_in,) or None if the hook never fires.
    """
    sum_sq = None  # type: Optional[torch.Tensor]

    def hook_fn(mod, inp, out):
        nonlocal sum_sq
        # inp[0]: (B, C_in) or (..., C_in); accumulate sum of squares on CPU
        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
        v = (x * x).sum(dim=0)
        sum_sq = v if sum_sq is None else (sum_sq + v)

    handle = model.get_submodule(layer_name).register_forward_hook(hook_fn)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in calib_dl:
            try:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                # Support both tokens and precomputed inputs_embeds (NLP), and pixel_values (Vision)
                model(**inputs) if inputs else None
            except Exception:
                # When pruning sequentially, some layers may be temporarily incompatible; skip safely.
                continue

    handle.remove()
    if sum_sq is None:
        return None
    return torch.sqrt(sum_sq)


def _mask_unstructured_topk(metric: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Per-output Top-K (Wanda): prune the smallest k = int(C_in * s) entries in each row.
    Returns boolean mask with True = prune.
    """
    C_out, C_in = metric.shape
    k = int(C_in * sparsity)
    if k <= 0:
        return torch.zeros_like(metric, dtype=torch.bool)
    vals, idx = torch.sort(metric, dim=1, stable=True)
    prune_sorted = torch.zeros_like(vals, dtype=torch.bool)
    prune_sorted[:, :k] = True
    mask = torch.zeros_like(metric, dtype=torch.bool)
    return mask.scatter(1, idx, prune_sorted)


def _mask_nm(metric: torch.Tensor, N: int, M: int) -> torch.Tensor:
    """
    N:M structured pruning per row: in each contiguous block of size M, prune (M-N) smallest.
    """
    assert 0 <= N <= M and M > 0, "Invalid N:M"
    C_out, C_in = metric.shape
    mask = torch.zeros_like(metric, dtype=torch.bool)
    if N == M:
        return mask
    drop = M - N
    for start in range(0, C_in, M):
        end = min(start + M, C_in)
        block = metric[:, start:end]
        if block.numel() == 0:
            continue
        k = min(drop, block.size(1))
        idx_local = torch.topk(block, k=k, dim=1, largest=False).indices
        mask.scatter_(1, idx_local + start, True)
    return mask


@torch.no_grad()
def wanda_prune(
    model: nn.Module,
    calib_dl: DataLoader,
    *,
    sparsity_ratio: float,
    device: torch.device,
    method: str = "wanda",  # "wanda" | "wanda_nm"
    nm_values: Optional[Tuple[int, int]] = None,  # e.g., (2,4)
) -> nn.Module:
    """
    One-shot Wanda pruning:
      score_ij = |w_ij| * ||X_j||_2  (per-output comparison, prune lowest s%)
    Structured N:M is supported via method == "wanda_nm".
    Returns a pruned *copy* of the given model.
    """
    print(">> Starting Wanda pruning ...")
    pruned = copy.deepcopy(model).to(device)
    targets = _layers_to_prune(pruned)

    for name, layer in tqdm(pruned.named_modules(), desc="Pruning layers"):
        if name not in targets:
            continue

        # 1) gather activation norms for input channels of this layer
        norms = _collect_act_norms(pruned, name, calib_dl, device)
        if norms is None:
            print(f"[WARN] No activations for {name}; skipping.")
            continue

        w = layer.weight.data
        metric = w.abs() * norms.to(w.device)  # (C_out, C_in)

        # 2) build prune mask
        if method == "wanda_nm" and nm_values:
            N, M = nm_values
            mask = _mask_nm(metric, N, M)
        else:
            mask = _mask_unstructured_topk(metric, sparsity_ratio)

        # 3) apply mask in-place
        w[mask.to(w.device)] = 0

    print(">> Wanda pruning finished.")
    return pruned
