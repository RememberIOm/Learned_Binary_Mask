# pruning_strategies.py
# -----------------------------------------------------------------------------
# Defines pruning strategies using the Strategy Pattern.
# Each class encapsulates a specific pruning algorithm (Wanda, LB-Mask, MI).
# This makes the pruning pipeline extensible and easy to maintain.
# -----------------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

# Import specific pruning algorithm implementations
from wanda_pruning import wanda_prune
from mi_pruning import mi_prune
from learned_binary_mask_pruning import (
    MaskPruneConfig,
    train_with_progressive_pruning,
    export_pruned_copy,
)
from learned_binary_mask_pruning import calc_sparsity  # A general utility


class PruningStrategy(ABC):
    """Abstract base class for a pruning strategy."""

    @abstractmethod
    def execute(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        sparsity_ratio: float,
        device: torch.device,
    ) -> tuple[torch.nn.Module, float]:
        """
        Executes the pruning strategy.
        Returns the pruned model and the achieved sparsity percentage.
        """
        pass


class WandaStrategy(PruningStrategy):
    """Implements the WANDA one-shot pruning algorithm."""

    def execute(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        sparsity_ratio: float,
        device: torch.device,
    ) -> tuple[torch.nn.Module, float]:
        pruned_model = wanda_prune(
            model,
            loader,
            sparsity_ratio=sparsity_ratio,
            device=device,
            method="wanda",
        )
        achieved_sparsity, _ = calc_sparsity(pruned_model)
        return pruned_model.to(device), achieved_sparsity


class MIStrategy(PruningStrategy):
    """Implements the Mutual Information based pruning algorithm."""

    def execute(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        sparsity_ratio: float,
        device: torch.device,
    ) -> tuple[torch.nn.Module, float]:
        pruned_model = mi_prune(
            model,
            loader,
            sparsity_ratio=sparsity_ratio,
            device=device,
        )
        achieved_sparsity, _ = calc_sparsity(pruned_model)
        return pruned_model.to(device), achieved_sparsity


class LBMaskStrategy(PruningStrategy):
    """Implements Learned Binary Mask (LB-Mask) pruning."""

    def execute(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        sparsity_ratio: float,
        device: torch.device,
    ) -> tuple[torch.nn.Module, float]:
        # Note: LB-Mask uses the loader for training the masks, not just calibration.
        mp_cfg = MaskPruneConfig(
            granularity="out",
            lr=5e-3,
            lmbda_l1=1e-3,
            num_epochs=3,
            skip_final_classifier=True,
            freeze_base=True,
            prune_during_train=True,
            schedule="constant",
            end_sparsity=sparsity_ratio,
            update_every=100,
            verbose=False,  # Keep console clean during sweeps
        )
        masked_model, overlays = train_with_progressive_pruning(
            base_model=model,
            train_dl=loader,
            mp_cfg=mp_cfg,
            base_lr=0.0,
            num_epochs=mp_cfg.num_epochs,
            device=device,
        )
        pruned_model = export_pruned_copy(model, overlays).to(device)
        achieved_sparsity, _ = calc_sparsity(pruned_model)
        return pruned_model, achieved_sparsity


# A factory to get the strategy instance based on a string identifier
def get_pruning_strategy(method: str) -> PruningStrategy:
    """Returns an instance of the requested pruning strategy."""
    if method == "wanda":
        return WandaStrategy()
    if method == "lbmask":
        return LBMaskStrategy()
    if method == "mi":
        return MIStrategy()
    raise ValueError(f"Unknown pruning method: {method}")
