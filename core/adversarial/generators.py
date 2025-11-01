# core/adversarial/generators.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from torch import nn

# ---------- Dataset primitives ----------


class DictSampleDataset(Dataset):
    """A lightweight Dataset where each item is a dict of tensors.
    Intended to carry per-sample inputs for generation/evaluation.

    Notes:
        - All tensors are expected to be on CPU. Move to device inside the loop.
        - Keys should match the model forward signature (e.g., 'inputs_embeds' or 'pixel_values', 'labels').
    """

    def __init__(self, items: Iterable[Dict[str, torch.Tensor]]):
        self._items = list(items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._items[idx]


# ---------- NLP: embedding conversion & adversarial generation ----------


def nlp_to_embeds_dataset(
    model: nn.Module,
    base_loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
) -> DictSampleDataset:
    """Convert token IDs to input embeddings for NLP models.

    Args:
        model: A model exposing an embedding layer (e.g., model.get_input_embeddings()).
        base_loader: Original DataLoader yielding dict with 'input_ids', attention masks, 'labels', etc.
        device: Torch device to run conversion.
    Returns:
        DictSampleDataset with 'inputs_embeds' (no 'input_ids') and corresponding labels/masks.
    """
    emb = model.get_input_embeddings()
    items = []
    model.eval()
    with torch.no_grad():
        for batch in base_loader:
            input_ids = batch["input_ids"].to(device)  # [B, T]
            embeds = emb(input_ids).detach().cpu()  # [B, T, D]
            item = {"inputs_embeds": embeds}
            # pass-through extra fields (attention_mask, labels, etc.)
            for k, v in batch.items():
                if k != "input_ids":
                    item[k] = v.cpu()
            # split into per-sample dicts
            for i in range(embeds.size(0)):
                items.append(
                    {
                        k: t[i] if t.dim() > 0 and t.size(0) == embeds.size(0) else t
                        for k, t in item.items()
                    }
                )
    return DictSampleDataset(items)


def nlp_adv_dataset(
    model: nn.Module,
    base_loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    eps_start: float,
    eps_max: float,
    eps_step: float,
    binary_target_idx: Optional[int] = None,
) -> DictSampleDataset:
    """Generate adversarial examples for NLP by perturbing embeddings (FGSM-like).

    Notes:
        - No fine-tuning: model params are NOT updated; gradients used only for crafting perturbations.
        - The target-vs-rest (binary) scenario is supported via `binary_target_idx` if provided.
    """
    assert eps_step > 0 and eps_start <= eps_max
    model.eval()
    items = []
    emb = model.get_input_embeddings()

    def _fgsm_step(embeds, labels, eps) -> torch.Tensor:
        embeds = embeds.clone().detach().requires_grad_(True)
        out = model(inputs_embeds=embeds, labels=labels)
        loss = out.loss
        loss.backward()
        # sign(grad) step w/o updating model weights
        adv = embeds + eps * embeds.grad.detach().sign()
        return adv.detach()

    with torch.enable_grad():
        for batch in base_loader:
            ids = batch.get("input_ids")
            if ids is not None:
                ids = ids.to(device)
                embeds = emb(ids)
            else:
                embeds = batch["inputs_embeds"].to(device)
            labels = batch["labels"].to(device)

            # optional relabeling for binary target-vs-rest
            if binary_target_idx is not None:
                labels = (labels == binary_target_idx).long()

            eps = eps_start
            while eps <= eps_max + 1e-12:
                adv_embeds = _fgsm_step(embeds, labels, eps)
                record = {k: v.cpu() for k, v in batch.items() if k != "input_ids"}
                record["inputs_embeds"] = adv_embeds.detach().cpu()
                record["labels"] = labels.detach().cpu()
                for i in range(adv_embeds.size(0)):
                    items.append(
                        {
                            kk: (
                                vv[i]
                                if torch.is_tensor(vv)
                                and vv.size(0) == adv_embeds.size(0)
                                else vv
                            )
                            for kk, vv in record.items()
                        }
                    )
                eps += eps_step
    return DictSampleDataset(items)


# ---------- Vision: adversarial generation ----------


def vision_adv_dataset(
    model: nn.Module,
    base_loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    eps_start_px: float,
    eps_max_px: float,
    eps_step_px: float,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    binary_target_idx: Optional[int] = None,
) -> DictSampleDataset:
    """Generate adversarial images (FGSM-like) in pixel space.

    Args:
        eps_*_px: Epsilon values in pixel space, expected in [0,1] scale.
        mean/std: Optional normalization stats; if provided, unnormalize before perturbation and re-normalize after.
    Notes:
        - No fine-tuning: model params are NOT updated.
    """
    assert eps_step_px > 0 and eps_start_px <= eps_max_px
    model.eval()
    items = []

    def _denorm(x):
        if mean is None or std is None:
            return x
        m = torch.tensor(mean, device=x.device)[None, :, None, None]
        s = torch.tensor(std, device=x.device)[None, :, None, None]
        return x * s + m

    def _norm(x):
        if mean is None or std is None:
            return x
        m = torch.tensor(mean, device=x.device)[None, :, None, None]
        s = torch.tensor(std, device=x.device)[None, :, None, None]
        return (x - m) / s

    def _fgsm_step(images, labels, eps) -> torch.Tensor:
        imgs = images.clone().detach().requires_grad_(True)
        out = model(pixel_values=_norm(imgs), labels=labels)
        loss = out.loss
        loss.backward()
        adv = imgs + eps * imgs.grad.detach().sign()
        adv = adv.clamp(0.0, 1.0)
        return adv.detach()

    with torch.enable_grad():
        for batch in base_loader:
            images = batch["pixel_values"].to(
                device
            )  # assume normalized if mean/std provided
            labels = batch["labels"].to(device)
            if binary_target_idx is not None:
                labels = (labels == binary_target_idx).long()

            imgs = _denorm(images)  # work in pixel space
            eps = eps_start_px
            while eps <= eps_max_px + 1e-12:
                adv_imgs = _fgsm_step(imgs, labels, eps)
                record = {k: v.cpu() for k, v in batch.items()}
                record["pixel_values"] = _norm(adv_imgs).cpu()
                record["labels"] = labels.cpu()
                for i in range(adv_imgs.size(0)):
                    items.append(
                        {
                            kk: (
                                vv[i]
                                if torch.is_tensor(vv)
                                and vv.size(0) == adv_imgs.size(0)
                                else vv
                            )
                            for kk, vv in record.items()
                        }
                    )
                eps += eps_step_px
    return DictSampleDataset(items)
