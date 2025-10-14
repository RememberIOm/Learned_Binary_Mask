# adv_utils.py
# -----------------------------------------------------------------------------
# Adversarial example utilities for NLP (embedding-level FGSM) and Vision (FGSM)
# Designed to work with Hugging Face classification heads used in main.py.
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Iterable, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


def map_target_vs_rest(labels: torch.Tensor, target_idx: int) -> torch.Tensor:
    """
    Map multiclass labels to binary with a single convention:
    0 = target class (labels == target_idx), 1 = negative (all others).
    """
    return (labels != int(target_idx)).long()


# ---------- Small dict-dataset wrapper ----------


class DictListDataset(Dataset):
    """Simple Dataset wrapper around a list of dict batches (per-sample dicts)."""

    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------- NLP: embedding-level helpers ----------


@torch.no_grad()
def _to_inputs_embeds_batch(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Convert token ids to static input embeddings without gradient."""
    emb_layer = model.get_input_embeddings()
    return emb_layer(input_ids)


@torch.no_grad()
def nlp_to_embeds_dataset(
    model,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> DictListDataset:
    """
    Convert a (tokenized) loader to an inputs_embeds-based dataset.
    This makes it collatable together with adversarial-embeds batches.
    """
    model.to(device)
    items: List[Dict] = []
    seen = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        embeds = _to_inputs_embeds_batch(model, input_ids)  # [B, T, H]
        for i in range(input_ids.size(0)):
            items.append(
                {
                    "inputs_embeds": embeds[i].cpu(),
                    "attention_mask": attn[i].cpu(),
                    "labels": labels[i].cpu(),
                }
            )
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    return DictListDataset(items)


def nlp_adv_dataset(
    model,
    data_loader,
    device,
    eps_start: float = 0.0,
    eps_max: float = 0.25,
    eps_step: float = 0.01,
    max_batches=None,
    binary_target_idx=None,
):
    """
    Build adversarial embeddings close to the decision boundary by epsilon sweep.
    For each sample, we detect the first step k where the prediction flips and
    append BOTH the 'just before' (k-1) and 'just after' (k) embeddings.

    If no flip occurs up to eps_max, we append a fallback pair at
    (eps_max - eps_step, eps_max).
    """
    model.eval()
    model.to(device)
    items = []
    seen = 0
    emb_layer = model.get_input_embeddings()

    assert eps_max >= eps_start >= 0.0, "Invalid epsilon range"
    assert eps_step > 0.0, "eps_step must be positive"
    num_steps = int(round((eps_max - eps_start) / eps_step))
    num_steps = max(num_steps, 1)

    def _forward_logits(embeds, attn_mask):
        if attn_mask is not None:
            return model(inputs_embeds=embeds, attention_mask=attn_mask).logits
        return model(inputs_embeds=embeds).logits

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask", None)
        attn = attn.to(device) if attn is not None else None
        y = batch["labels"].to(device)

        # Labels for gradient (binary map if requested)
        y_fgsm = (
            map_target_vs_rest(y, int(binary_target_idx))
            if binary_target_idx is not None
            else y
        )

        # Gradient direction at epsilon=0
        embeds0 = emb_layer(input_ids).detach().clone().requires_grad_(True)
        out0 = model(inputs_embeds=embeds0, attention_mask=attn, labels=y_fgsm)
        loss0 = out0.loss
        model.zero_grad(set_to_none=True)
        loss0.backward()
        direction = embeds0.grad.detach().sign()

        with torch.no_grad():
            base_pred = out0.logits.argmax(dim=-1)

        # Track which samples already found their crossing
        done = torch.zeros_like(base_pred, dtype=torch.bool)

        for k in range(1, num_steps + 1):
            eps_prev = eps_start + (k - 1) * eps_step
            eps_curr = eps_start + k * eps_step

            adv_prev = (embeds0 + eps_prev * direction).detach()
            adv_curr = (embeds0 + eps_curr * direction).detach()

            with torch.no_grad():
                pred_prev = _forward_logits(adv_prev, attn).argmax(dim=-1)
                pred_curr = _forward_logits(adv_curr, attn).argmax(dim=-1)

            cross = (~done) & (pred_prev.eq(base_pred)) & (~pred_curr.eq(base_pred))
            idx = torch.nonzero(cross).view(-1)

            if idx.numel():
                for i in idx.tolist():
                    # Option: use midpoint -> 0.5*(adv_prev[i]+adv_curr[i])
                    items.append(
                        {
                            "inputs_embeds": adv_prev[i].cpu(),
                            "attention_mask": (
                                attn[i].cpu() if attn is not None else None
                            ),
                            "labels": y[i].cpu(),
                        }
                    )
                    items.append(
                        {
                            "inputs_embeds": adv_curr[i].cpu(),
                            "attention_mask": (
                                attn[i].cpu() if attn is not None else None
                            ),
                            "labels": y[i].cpu(),
                        }
                    )
                done[idx] = True

            if torch.all(done):
                break

        # Fallback for samples that never flipped up to eps_max
        if not torch.all(done):
            eps_prev = max(eps_start, eps_max - eps_step)
            eps_curr = eps_max
            adv_prev = (embeds0 + eps_prev * direction).detach()
            adv_curr = (embeds0 + eps_curr * direction).detach()
            remain = torch.nonzero(~done).view(-1)
            for i in remain.tolist():
                items.append(
                    {
                        "inputs_embeds": adv_prev[i].cpu(),
                        "attention_mask": (attn[i].cpu() if attn is not None else None),
                        "labels": y[i].cpu(),
                    }
                )
                items.append(
                    {
                        "inputs_embeds": adv_curr[i].cpu(),
                        "attention_mask": (attn[i].cpu() if attn is not None else None),
                        "labels": y[i].cpu(),
                    }
                )

        seen += 1
        if max_batches is not None and seen >= max_batches:
            break

    return DictListDataset(items)


# ---------- Vision: pixel-level FGSM ----------


def vision_adv_dataset(
    model,
    data_loader,
    device,
    eps_start_px: float = 0.0,
    eps_max_px: float = 8 / 255,
    eps_step_px: float = 2 / 255,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    max_batches=None,
    binary_target_idx=None,
):
    """
    Build adversarial images near the decision boundary via epsilon sweep (FGSM direction).
    ...
    """
    model.eval()
    model.to(device)
    items = []
    seen = 0

    assert eps_max_px >= eps_start_px >= 0.0, "Invalid epsilon range"
    assert eps_step_px > 0.0, "eps_step_px must be positive"
    num_steps = int(round((eps_max_px - eps_start_px) / eps_step_px))

    # Prepare normalization tensors if provided
    mean_t = std_t = min_val = max_val = None
    if (mean is not None) and (std is not None):
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(
            1, -1, 1, 1
        )
        std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        min_val = (0.0 - mean_t) / std_t
        max_val = (1.0 - mean_t) / std_t

    for batch in data_loader:
        x = batch["pixel_values"].to(device).detach().clone().requires_grad_(True)
        y = batch["labels"].to(device)
        y_fgsm = (
            map_target_vs_rest(y, int(binary_target_idx))
            if binary_target_idx is not None
            else y
        )

        # Gradient direction at epsilon=0
        out = model(pixel_values=x, labels=y_fgsm)
        loss = out.loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        direction = x.grad.detach().sign()

        # Clean prediction
        with torch.no_grad():
            base_logits = out.logits
            base_pred = base_logits.argmax(dim=-1)

        # Sweep epsilon and find the first change
        selected = None
        for k in range(1, num_steps + 1):
            eps_curr_px = eps_start_px + k * eps_step_px
            eps_prev_px = eps_curr_px - eps_step_px

            if std_t is not None:
                step_curr = eps_curr_px / std_t
                step_prev = eps_prev_px / std_t
            else:
                step_curr = torch.as_tensor(eps_curr_px, device=x.device, dtype=x.dtype)
                step_prev = torch.as_tensor(eps_prev_px, device=x.device, dtype=x.dtype)

            adv_prev = x + step_prev * direction
            adv_curr = x + step_curr * direction

            if (min_val is not None) and (max_val is not None):
                adv_prev = torch.maximum(torch.minimum(adv_prev, max_val), min_val)
                adv_curr = torch.maximum(torch.minimum(adv_curr, max_val), min_val)

            with torch.no_grad():
                pred_curr = model(pixel_values=adv_curr).logits.argmax(dim=-1)

            if not torch.equal(pred_curr, base_pred):
                selected = adv_curr  # or 0.5 * (adv_prev + adv_curr) for mid-point
                break

        if selected is None:
            # Fallback: take the largest epsilon if no change occurred
            if std_t is not None:
                step_max = eps_max_px / std_t
            else:
                step_max = torch.as_tensor(eps_max_px, device=x.device, dtype=x.dtype)
            selected = x + step_max * direction
            if (min_val is not None) and (max_val is not None):
                selected = torch.maximum(torch.minimum(selected, max_val), min_val)

        selected = selected.detach()
        for i in range(x.size(0)):
            items.append({"pixel_values": selected[i].cpu(), "labels": y[i].cpu()})

        seen += 1
        if max_batches is not None and seen >= max_batches:
            break

    return DictListDataset(items)


# ---------- Mixing (original + adversarial) ----------


def _sample_indices(n: int, k: int) -> torch.Tensor:
    k = max(0, min(k, n))
    if k == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.randperm(n)[:k]


def make_mixed_loader(
    *,
    base_ds: DictListDataset,
    adv_ds: DictListDataset,
    adv_ratio: float,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader that mixes base_ds and adv_ds with given adv_ratio in [0,1].
    For NLP, base_ds should ALSO be in inputs_embeds form (not input_ids).
    """
    n = len(base_ds)
    n_adv = int(round(n * float(adv_ratio)))
    n_base = n - n_adv

    # Subsample both sides to match requested mix
    idx_base = _sample_indices(len(base_ds), n_base)
    idx_adv = _sample_indices(len(adv_ds), n_adv)

    merged: List[Dict] = []
    for i in idx_base.tolist():
        merged.append(base_ds.items[i])
    for j in idx_adv.tolist():
        merged.append(adv_ds.items[j])

    # If one side is too small, pad from the other to keep size ~n
    while len(merged) < n and len(adv_ds) > 0:
        merged.append(adv_ds.items[int(torch.randint(0, len(adv_ds), (1,)).item())])
    while len(merged) < n and len(base_ds) > 0:
        merged.append(base_ds.items[int(torch.randint(0, len(base_ds), (1,)).item())])

    mixed = DictListDataset(merged)
    return DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------- Evaluation on adversarial test sets ----------


@torch.no_grad()
def evaluate_adv_accuracy(
    model,
    adv_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Generic accuracy on adversarial loader.
    Works for loaders that emit either:
      - {inputs_embeds, attention_mask, labels}  (NLP)
      - {pixel_values, labels}                  (Vision)
    """
    model.eval()
    model.to(device)
    total = 0
    correct = 0

    for batch in adv_loader:
        if "pixel_values" in batch:
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            logits = model(pixel_values=x).logits
        else:
            x = batch["inputs_embeds"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(inputs_embeds=x, attention_mask=attn).logits

        pred = logits.argmax(dim=-1)
        total += y.numel()
        correct += int((pred == y).sum().item())

    return correct / max(1, total)
