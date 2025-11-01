# core/utils/adv_utils.py
import torch


def map_target_vs_rest(labels: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Map multiclass labels -> binary {0: target, 1: negative}."""
    return (labels != int(target_idx)).long()
