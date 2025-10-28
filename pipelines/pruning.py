# pipelines/pruning.py

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from pruning_strategies import get_pruning_strategy


def prune_model(
    base_model: torch.nn.Module,
    *,
    method: str,
    loader: DataLoader,
    device: torch.device,
    sparsity: float,
) -> tuple[torch.nn.Module, float]:
    """
    Selects and executes a pruning strategy.

    This function acts as a client for the Strategy Pattern. It fetches the
    appropriate strategy object based on the 'method' string and delegates
    the pruning task to it.

    Returns:
        A tuple containing the pruned model and the achieved sparsity percentage.
    """
    # Get the strategy object from the factory
    strategy = get_pruning_strategy(method)

    # Execute the strategy
    pruned_model, achieved_sparsity = strategy.execute(
        model=base_model,
        loader=loader,
        sparsity_ratio=sparsity,
        device=device,
    )

    return pruned_model, achieved_sparsity
