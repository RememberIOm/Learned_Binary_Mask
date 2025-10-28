# pipelines/adv_data.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from config import ExpConfig
from data_utils import is_vision
from adv_utils import (
    nlp_to_embeds_dataset,
    nlp_adv_dataset,
    vision_adv_dataset,
    make_mixed_loader,
    DictListDataset,
)


def build_mixed_pruning_loader(
    model,
    cfg: ExpConfig,
    base_loader: DataLoader,
    device: torch.device,
    adv_ratio: float,
    *,
    eps_nlp: dict,
    eps_vision: dict,
    vision_stats: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
) -> DataLoader:
    """Return a calibration/train loader mixing original and adversarial samples."""
    if is_vision(cfg):
        items = []
        for batch in base_loader:
            px, y = batch["pixel_values"], batch["labels"]
            for i in range(px.size(0)):
                items.append({"pixel_values": px[i], "labels": y[i]})
        base_ds = DictListDataset(items)
        mean, std = vision_stats or (None, None)
        adv_ds = vision_adv_dataset(
            model,
            base_loader,
            device,
            eps_start_px=eps_vision["start"],
            eps_max_px=eps_vision["max"],
            eps_step_px=eps_vision["step"],
            mean=mean,
            std=std,
        )
    else:
        base_ds = nlp_to_embeds_dataset(model, base_loader, device)
        adv_ds = nlp_adv_dataset(
            model,
            base_loader,
            device,
            eps_start=eps_nlp["start"],
            eps_max=eps_nlp["max"],
            eps_step=eps_nlp["step"],
        )
    return make_mixed_loader(
        base_ds=base_ds, adv_ds=adv_ds, adv_ratio=adv_ratio, batch_size=8, shuffle=True
    )


def build_adversarial_test_loader(
    model,
    test_loader: DataLoader,
    cfg: ExpConfig,
    device: torch.device,
    *,
    eps_nlp: dict,
    eps_vision: dict,
    vision_stats: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
    binary_target_idx: Optional[int] = None,
) -> DataLoader:
    """Return a DataLoader consisting only of adversarial samples for robustness eval."""
    if is_vision(cfg):
        mean, std = vision_stats or (None, None)
        adv_ds = vision_adv_dataset(
            model,
            test_loader,
            device,
            eps_start_px=eps_vision["start"],
            eps_max_px=eps_vision["max"],
            eps_step_px=eps_vision["step"],
            mean=mean,
            std=std,
            binary_target_idx=binary_target_idx,
        )
    else:
        adv_ds = nlp_adv_dataset(
            model,
            test_loader,
            device,
            eps_start=eps_nlp["start"],
            eps_max=eps_nlp["max"],
            eps_step=eps_nlp["step"],
            binary_target_idx=binary_target_idx,
        )
    return DataLoader(adv_ds, batch_size=cfg.batch_size)
