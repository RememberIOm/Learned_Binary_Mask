# decomposition.py
# ---------------------------------------------------------------------
# Turn a multiclass Linear head into TWO binary classifiers:
#   - "target" head for the chosen class index
#   - "negative" head for the complement (others aggregated)
# The original model is DEEP-COPIED; base weights are not modified.
# Works for models whose final head is a plain nn.Linear named
#   one of {"classifier", "classification_head", "fc"}.
# ---------------------------------------------------------------------

from __future__ import annotations
from typing import Optional
import copy
import torch
import torch.nn as nn


_FINAL_HEAD_NAMES = ("classifier", "classification_head", "fc")


class TargetVsRestLinear(nn.Module):
    """Two parallel Linear heads producing logits [target, negative]."""

    def __init__(self, base_linear: nn.Linear, target_idx: int):
        super().__init__()
        in_f = int(base_linear.in_features)
        out_f = int(base_linear.out_features)
        assert 0 <= target_idx < out_f, "target_idx is out of range"

        use_bias = base_linear.bias is not None
        self.pos = nn.Linear(in_f, 1, bias=use_bias)
        self.neg = nn.Linear(in_f, 1, bias=use_bias)

        # Initialize from the original multiclass head.
        with torch.no_grad():
            # target head := the row for target class
            self.pos.weight.copy_(base_linear.weight[target_idx : target_idx + 1])
            if use_bias:
                self.pos.bias.copy_(base_linear.bias[target_idx : target_idx + 1])

            # negative head := mean of all non-target rows (simple, stable init)
            neg_idx = [i for i in range(out_f) if i != target_idx]
            w_neg = base_linear.weight[neg_idx].mean(dim=0, keepdim=True)
            self.neg.weight.copy_(w_neg)
            if use_bias:
                b_neg = base_linear.bias[neg_idx].mean().view(1)
                self.neg.bias.copy_(b_neg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: [B, 2] -> [:, 0] = target, [:, 1] = negative
        logit_pos = self.pos(x)
        logit_neg = self.neg(x)
        return torch.cat([logit_pos, logit_neg], dim=-1)


def decompose_to_target_vs_rest(model: nn.Module, target_idx: int) -> nn.Module:
    """
    Return a prunable copy whose final head outputs [target, negative] logits.
    Also sync HF num_labels to 2 so that loss/reshapes are correct.
    """
    m = copy.deepcopy(model)
    for name in _FINAL_HEAD_NAMES:
        if hasattr(m, name):
            head = getattr(m, name)
            if isinstance(head, nn.Linear):
                setattr(m, name, TargetVsRestLinear(head, target_idx))
                # --- Ensure HF loss uses correct label dimension (2) ---
                if hasattr(m, "num_labels"):
                    m.num_labels = 2
                if hasattr(m, "config") and hasattr(m.config, "num_labels"):
                    m.config.num_labels = 2
                # Optional: stamp target for downstream helpers
                setattr(m, "_binary_target_idx", int(target_idx))
                return m
    raise ValueError(
        "Expected a plain nn.Linear final head named one of "
        f"{_FINAL_HEAD_NAMES}, but none was found."
    )
