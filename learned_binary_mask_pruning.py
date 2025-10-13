# learned_binary_mask_pruning.py
# ---------------------------------------------------------------------
# Learnable binary-mask pruning for Linear layers
# - Freeze original weights; learn a same-shaped mask layer (logits)
# - Encourage sparsity via L1 on mask probabilities
# - Project to binary and export a pruned copy without touching original
# ---------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


Granularity = Literal["weight", "out"]
ScheduleType = Literal["constant", "linear", "cosine"]


@dataclass
class MaskPruneConfig:
    # Mask learning
    granularity: Granularity = "out"  # "out" (per-neuron) or "weight" (per-weight)
    init_keep_prob: float = 0.99  # initial keep prob for masks
    temperature: float = 1.0  # sigmoid temperature
    use_hard_ste: bool = True  # straight-through for hard binarization during train

    # Optimization
    lr: float = 5e-3
    weight_decay: float = 0.0
    num_epochs: int = 3
    max_steps: Optional[int] = None
    lmbda_l1: float = 1e-3  # sparsity regularizer weight (L1 on mask probs)

    # Targets / policy
    target_sparsity: float = 0.5  # target (best-effort via L1 pressure)
    threshold: float = 0.5  # binarization threshold at projection
    skip_final_classifier: bool = True  # skip typical classifier heads

    # Joint-training & Schedule options
    freeze_base: bool = True  # False => train base weights jointly
    prune_during_train: bool = False  # True => do pruning during training
    schedule: ScheduleType = "constant"  # constant | linear | cosine
    start_sparsity: float = 0.0  # schedule start ratio
    end_sparsity: float = 0.5  # schedule end (or constant) ratio
    begin_step: int = 0  # schedule starts at this global step
    end_step: int = 10000  # schedule ends at this step
    update_every: int = 100  # how often to recompute threshold
    hard_apply: bool = False  # permanently zero weights on updates
    zero_bias_when_masked: bool = True  # also zero bias for masked rows

    # Misc
    log_every: int = 100
    verbose: bool = True


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _final_head_names(model: nn.Module) -> set[str]:
    keys = {"classifier", "classification_head", "fc"}
    out = set()
    for k in keys:
        if hasattr(model, k):
            out.add(k)
    return out


class LinearWithLearnedMask(nn.Module):
    """
    Wrapper that holds a Linear 'base' (trainable or frozen) and a learnable mask.
    Forward uses (W ⊙ M). Threshold can be overridden per-step for scheduled pruning.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        name_in_model: str,
        granularity: Granularity = "out",
        init_keep_prob: float = 0.99,
        temperature: float = 1.0,
        use_hard_ste: bool = True,
        device: Optional[torch.device] = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.name_in_model = name_in_model
        self.granularity = granularity
        self.temperature = temperature
        self.use_hard_ste = use_hard_ste
        self.override_threshold: Optional[float] = None

        # Make a frozen copy of the original Linear
        self.base = nn.Linear(
            base_linear.in_features,
            base_linear.out_features,
            bias=base_linear.bias is not None,
        )
        self.base.load_state_dict(base_linear.state_dict())
        for p in self.base.parameters():
            p.requires_grad = not freeze_base

        # Create learnable mask logits
        if granularity == "weight":
            mask_shape = (self.base.out_features, self.base.in_features)
        elif granularity == "out":
            mask_shape = (self.base.out_features, 1)
        else:
            raise ValueError(f"Unknown granularity: {granularity}")

        init_logit = _logit(init_keep_prob)
        self.mask_logits = nn.Parameter(torch.full(mask_shape, init_logit))

        # Binary snapshot (for hard projection when requested)
        self.register_buffer("mask_binary", torch.empty(0), persistent=False)
        self.use_binary = False

        if device is not None:
            self.to(device)

    # --------- utilities ---------
    def set_threshold(self, thr: Optional[float]):
        """Override the binarization threshold used in current_mask()."""
        self.override_threshold = thr

    def mask_probs(self) -> torch.Tensor:
        """Sigmoid temperature-scaled probabilities in [0,1]."""
        return torch.sigmoid(self.mask_logits / self.temperature)

    def regularization_loss(self) -> torch.Tensor:
        """
        L1 on mask probabilities encourages sparsity (smaller probs -> more zeros).
        """
        return self.mask_probs().mean()

    def binarize(self, threshold: float = 0.5):
        """Project probabilities to {0,1} and store as a buffer."""
        probs = self.mask_probs()
        with torch.no_grad():
            self.mask_binary = (probs > threshold).float()
        self.use_binary = True

    def current_mask(self) -> torch.Tensor:
        """
        Returns the mask tensor used in forward:
        - if projected: binary mask
        - else: soft or hard mask with STE
        """
        if self.use_binary and self.mask_binary.numel() > 0:
            return self.mask_binary

        probs = self.mask_probs()
        thr = 0.5 if self.override_threshold is None else float(self.override_threshold)

        if not self.use_hard_ste:
            # Soft mask (no binarization)
            return probs

        # Hard mask with straight-through estimator (uses dynamic thr)
        hard = (probs > thr).float()
        return hard + probs - probs.detach()

    def effective_row_mask(self) -> torch.Tensor:
        """
        Returns a length-[out_features] vector between 0 and 1
        (used to gate bias and report neuron sparsity).
        """
        m = self.current_mask()
        if self.granularity == "out":
            return m.view(-1)
        else:  # weight-wise: mean of row
            return m.mean(dim=1)

    def weight_mask(self) -> torch.Tensor:
        """Return mask broadcast to weight shape [out, in]."""
        m = self.current_mask()
        if self.granularity == "out":
            return m.expand(self.base.out_features, self.base.in_features)
        return m

    @torch.no_grad()
    def apply_hard_prune(
        self, threshold: Optional[float] = None, zero_bias: bool = True
    ):
        """
        Permanently zero out weights/bias based on current (or given) threshold.
        This is used for progressive *hard* pruning during training.
        """
        thr = (
            float(threshold)
            if threshold is not None
            else (
                0.5
                if self.override_threshold is None
                else float(self.override_threshold)
            )
        )
        probs = self.mask_probs()
        if self.granularity == "out":
            m_bin = (probs > thr).float()  # [out, 1]
            M = m_bin.expand(self.base.out_features, self.base.in_features).to(
                self.base.weight.device
            )
            row_m = m_bin.view(-1).to(self.base.weight.device)
        else:
            M = (probs > thr).float().to(self.base.weight.device)
            row_m = M.mean(dim=1)

        self.base.weight.data.mul_(M)
        if zero_bias and self.base.bias is not None:
            self.base.bias.data.mul_(row_m)

    # --------- forward ---------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.base.weight
        B = self.base.bias
        M = self.weight_mask()
        W_eff = W * M
        if x.device != W.device:
            x = x.to(W.device, non_blocking=True)

        if B is not None:
            row_m = self.effective_row_mask()
            B_eff = B * row_m
        else:
            B_eff = None

        return F.linear(x, W_eff, B_eff)


# ---------------- construction / wrapping ----------------


def _replace_linears_with_masks(
    module: nn.Module,
    overlays: Dict[str, LinearWithLearnedMask],
    prefix: str,
    cfg: MaskPruneConfig,
    device: Optional[torch.device],
    skip_names: set[str],
):
    """
    In-place replace Linear children with LinearWithLearnedMask, recursively.
    """
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear) and not any(
            part in skip_names for part in full_name.split(".")
        ):
            masked = LinearWithLearnedMask(
                base_linear=child,
                name_in_model=full_name,
                granularity=cfg.granularity,
                init_keep_prob=cfg.init_keep_prob,
                temperature=cfg.temperature,
                use_hard_ste=cfg.use_hard_ste,
                device=device,
                freeze_base=cfg.freeze_base,
            )
            setattr(module, child_name, masked)
            overlays[full_name] = masked
        else:
            _replace_linears_with_masks(
                child, overlays, full_name, cfg, device, skip_names
            )


def build_masked_model_from(
    base_model: nn.Module,
    cfg: MaskPruneConfig,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, LinearWithLearnedMask]]:
    """
    Create a deep-copied model where all Linear layers are replaced with
    LinearWithLearnedMask. The original base_model is untouched.
    Returns (masked_model, overlays).
    """
    masked_model = copy.deepcopy(base_model)
    overlays: Dict[str, LinearWithLearnedMask] = {}

    skip_names = _final_head_names(masked_model) if cfg.skip_final_classifier else set()
    _replace_linears_with_masks(
        masked_model, overlays, prefix="", cfg=cfg, device=device, skip_names=skip_names
    )

    if device is None:
        try:
            device = next(base_model.parameters()).device
        except StopIteration:
            device = None
    if device is not None:
        masked_model.to(device)
    return masked_model, overlays


# ---------------- training / projection / export ----------------


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def _forward_for_hf_like(model: nn.Module, batch: dict):
    """
    Works for both HF NLP (expects input_ids/attention_mask/labels) and
    HF ViT-style vision (pixel_values/labels). Keeps labels in batch.
    """
    if "pixel_values" in batch:
        return model(pixel_values=batch["pixel_values"], labels=batch.get("labels"))
    else:
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        return model(**inputs, labels=batch.get("labels"))


def train_masks(
    masked_model: nn.Module,
    train_dl: DataLoader,
    cfg: MaskPruneConfig,
    device: torch.device,
):
    """
    Train only mask logits to preserve accuracy while promoting sparsity.
    Original weights in masked_model are frozen by construction.
    """
    masked_model.train()
    # Only optimize mask parameters
    params = [p for n, p in masked_model.named_parameters() if "mask_logits" in n]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step = 0
    pbar = tqdm(
        total=cfg.max_steps or (cfg.num_epochs * len(train_dl)),
        desc="TrainMasks",
        leave=False,
    )

    for epoch in range(1, cfg.num_epochs + 1):
        for batch in train_dl:
            if cfg.max_steps and global_step >= cfg.max_steps:
                break
            batch = _to_device(batch, device)
            out = _forward_for_hf_like(masked_model, batch)
            loss_main = (
                out.loss if hasattr(out, "loss") and out.loss is not None else 0.0
            )

            # L1 sparsity regularizer over all overlays
            l1_terms = [
                m.regularization_loss()
                for m in masked_model.modules()
                if isinstance(m, LinearWithLearnedMask)
            ]
            l1 = (
                torch.stack(l1_terms).mean().to(device)
                if l1_terms
                else torch.tensor(0.0, device=device)
            )
            loss = loss_main + cfg.lmbda_l1 * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.update(1)
            if cfg.verbose and global_step % cfg.log_every == 0:
                with torch.no_grad():
                    overall_keep = []
                    for m in masked_model.modules():
                        if isinstance(m, LinearWithLearnedMask):
                            overall_keep.append(m.effective_row_mask().mean().item())
                    keep_mean = float(sum(overall_keep) / max(1, len(overall_keep)))
                print(
                    f"[mask-train] step={global_step} loss={float(loss):.4f} "
                    f"main={float(loss_main):.4f} l1={float(l1):.4f} "
                    f"avg_keep≈{keep_mean:.3f} (target keep≈{1.0 - cfg.target_sparsity:.3f})"
                )
        if cfg.max_steps and global_step >= cfg.max_steps:
            break
    pbar.close()


@torch.no_grad()
def project_masks_to_binary(
    masked_model: nn.Module,
    overlays: Dict[str, LinearWithLearnedMask],
    threshold: Optional[float] = None,
):
    """
    Binarize learned masks to {0,1} (hard pruning). Keeps masks inside overlays.
    """
    thr = threshold if threshold is not None else 0.5
    for mod in overlays.values():
        mod.binarize(thr)


@torch.no_grad()
def export_pruned_copy(
    base_model: nn.Module,
    overlays: Dict[str, LinearWithLearnedMask],
) -> nn.Module:
    """
    Return a deep-copied vanilla model whose Linear weights/biases are multiplied
    by the overlays' binary masks. The base_model remains unchanged.
    """
    pruned = copy.deepcopy(base_model)

    for name, ov in overlays.items():
        # Find corresponding Linear in the pruned copy
        target = pruned.get_submodule(name)
        if not isinstance(target, nn.Linear):
            # If the structure differs (e.g., HF head replaced earlier), skip safely
            continue

        # Weight mask to full [out, in]
        if ov.use_binary and ov.mask_binary.numel() > 0:
            m = ov.mask_binary
        else:
            m = (ov.mask_probs() > 0.5).float()  # fallback

        if ov.granularity == "out":
            M = m.expand(target.out_features, target.in_features).to(
                target.weight.device
            )
            row_m = m.view(-1).to(target.weight.device)
        else:
            M = m.to(target.weight.device)
            row_m = m.mean(dim=1).to(target.weight.device)

        target.weight.data = target.weight.data * M
        if target.bias is not None:
            target.bias.data = target.bias.data * row_m

    return pruned


@torch.no_grad()
def _overall_sparsity_if_threshold(
    base_model: nn.Module,
    overlays: Dict[str, LinearWithLearnedMask],
    threshold: float,
) -> float:
    """
    Return overall sparsity ratio (0..1) across ALL Linear weights in base_model
    if overlays are binarized with the given threshold.
    - Overlaid Linear layers: zeros come from base zeros OR masked zeros.
    - Non-overlaid Linear layers: only inherent base zeros are counted.
    """
    total_elems = 0
    total_zeros = 0

    # 1) Overlaid linears: use frozen copies inside overlays
    for name, ov in overlays.items():
        W = ov.base.weight.detach().to("cpu")
        # Build binary mask at this threshold
        probs = ov.mask_probs().detach().to("cpu")
        if ov.granularity == "out":
            m_bin = probs > threshold  # shape [out, 1] boolean
            M = m_bin.expand(ov.base.out_features, ov.base.in_features)
        else:  # "weight"
            M = probs > threshold

        # Final zeros are where either W==0 or M==0
        W_nz = W != 0
        M_nz = M != 0
        nonzero = (W_nz & M_nz).sum().item()
        zeros = W.numel() - nonzero

        total_elems += W.numel()
        total_zeros += zeros

    # 2) Non-overlaid linears in base_model remain as-is
    over_names = set(overlays.keys())
    for name, m in base_model.named_modules():
        if isinstance(m, nn.Linear) and name not in over_names:
            W = m.weight.detach().to("cpu")
            total_elems += W.numel()
            total_zeros += int((W == 0).sum().item())

    return (total_zeros / total_elems) if total_elems else 0.0


@torch.no_grad()
def line_search_threshold_for_target(
    base_model: nn.Module,
    overlays: Dict[str, LinearWithLearnedMask],
    target_sparsity: float,
    tol: float = 1e-3,  # tolerance on sparsity ratio (e.g., 0.001 == 0.1pp)
    max_iter: int = 25,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Binary-search a global threshold t in [0,1] so that the overall sparsity
    across ALL Linear weights is close to target_sparsity (ratio in 0..1).
    Returns (best_threshold, achieved_sparsity_ratio).
    """
    # Bounds
    lo, hi = 0.0, 1.0
    s_lo = _overall_sparsity_if_threshold(base_model, overlays, lo)
    s_hi = _overall_sparsity_if_threshold(base_model, overlays, hi)

    if verbose:
        print(
            f"[line-search] min={s_lo*100:.3f}%  max={s_hi*100:.3f}%  target={target_sparsity*100:.3f}%"
        )

    # If target is outside achievable range, clamp to nearest bound
    if target_sparsity <= s_lo + tol:
        if verbose:
            print(f"[line-search] Target below min; using threshold={lo:.6f}")
        return lo, s_lo
    if target_sparsity >= s_hi - tol:
        if verbose:
            print(f"[line-search] Target above max; using threshold={hi:.6f}")
        return hi, s_hi

    # Monotone bisection
    best_t, best_s = None, None
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        s_mid = _overall_sparsity_if_threshold(base_model, overlays, mid)
        if verbose:
            print(f"[line-search] it={it:02d} thr={mid:.6f} sparsity={s_mid*100:.3f}%")

        # Keep track of the best-so-far (closest to target)
        if (best_s is None) or (
            abs(s_mid - target_sparsity) < abs(best_s - target_sparsity)
        ):
            best_t, best_s = mid, s_mid

        if abs(s_mid - target_sparsity) <= tol:
            return mid, s_mid
        if s_mid < target_sparsity:
            lo = mid
        else:
            hi = mid

    # Fall back to closest mid if tolerance not met within max_iter
    return best_t, best_s


@torch.no_grad()
def project_by_line_search(
    masked_model: nn.Module,
    base_model: nn.Module,
    overlays: Dict[str, LinearWithLearnedMask],
    target_sparsity: float,
    tol: float = 1e-3,
    max_iter: int = 25,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Convenience: find threshold via line search, then project overlays to binary
    with that threshold. Returns (threshold, achieved_sparsity_ratio).
    """
    thr, achieved = line_search_threshold_for_target(
        base_model=base_model,
        overlays=overlays,
        target_sparsity=target_sparsity,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )
    project_masks_to_binary(masked_model, overlays, threshold=thr)
    return thr, achieved


# ---------------- progressive pruning during training ----------------


class SparsityScheduler:
    """Time-based sparsity schedule."""

    def __init__(
        self,
        start: float,
        end: float,
        t0: int,
        t1: int,
        kind: ScheduleType = "constant",
    ):
        assert 0.0 <= start <= 1.0 and 0.0 <= end <= 1.0
        assert t1 >= t0, "end_step must be >= begin_step"
        self.start, self.end = float(start), float(end)
        self.t0, self.t1 = int(t0), int(t1)
        self.kind = kind

    def value(self, step: int) -> float:
        if step <= self.t0:
            return self.start if self.kind != "constant" else self.end
        if step >= self.t1:
            return self.end
        # progress in [0,1]
        u = (step - self.t0) / max(1, (self.t1 - self.t0))
        if self.kind == "linear":
            return self.start + u * (self.end - self.start)
        if self.kind == "cosine":
            # smooth monotone rise from start -> end
            import math

            return self.end + (self.start - self.end) * 0.5 * (
                1.0 + math.cos(math.pi * u)
            )
        # constant
        return self.end


class ProgressivePruner:
    """
    Maintains a target sparsity schedule during training by:
      - solving for a global threshold via line-search, and
      - setting each overlay's dynamic threshold accordingly.
      - optionally performing hard pruning on a cadence (cfg.hard_apply).
    """

    def __init__(
        self,
        masked_model: nn.Module,
        overlays: Dict[str, LinearWithLearnedMask],
        cfg: MaskPruneConfig,
        device: torch.device,
        total_steps_hint: Optional[int] = None,
        base_ref_model: Optional[nn.Module] = None,
    ):
        self.model = masked_model
        self.overlays = overlays
        self.cfg = cfg
        self.device = device
        self.base_ref_model = base_ref_model or masked_model
        self.sched = SparsityScheduler(
            start=cfg.start_sparsity,
            end=cfg.end_sparsity,
            t0=cfg.begin_step,
            t1=(
                cfg.end_step
                if cfg.end_step is not None
                else (total_steps_hint or 10000)
            ),
            kind=cfg.schedule,
        )
        self.last_thr: Optional[float] = None

    @torch.no_grad()
    def _set_threshold_all(self, thr: float):
        for m in self.overlays.values():
            m.set_threshold(thr)

    @torch.no_grad()
    def _maybe_hard_apply(self, step: int):
        if not self.cfg.hard_apply:
            return
        if (step % max(1, self.cfg.update_every) == 0) or (step >= self.sched.t1):
            for m in self.overlays.values():
                m.apply_hard_prune(
                    self.last_thr, zero_bias=self.cfg.zero_bias_when_masked
                )

    @torch.no_grad()
    def on_step(self, step: int):
        if step % max(1, self.cfg.update_every) != 0 and self.last_thr is not None:
            return  # keep current threshold until next update

        target = float(self.sched.value(step))
        # Compute a global threshold that yields target sparsity across ALL Linear params
        thr, _ = line_search_threshold_for_target(
            base_model=self.base_ref_model,
            overlays=self.overlays,
            target_sparsity=target,
            tol=1e-3,
            max_iter=25,
            verbose=False,
        )
        self.last_thr = float(thr)
        self._set_threshold_all(self.last_thr)
        self._maybe_hard_apply(step)


def train_with_progressive_pruning(
    base_model: nn.Module,
    train_dl: DataLoader,
    mp_cfg: MaskPruneConfig,
    *,
    base_lr: float,
    num_epochs: int,
    device: torch.device,
):
    """
    Jointly train base weights (if cfg.freeze_base=False) and mask logits while
    enforcing a sparsity schedule during training. Returns (masked_model, overlays).
    """
    masked_model, overlays = build_masked_model_from(base_model, mp_cfg, device=device)
    masked_model.train()

    # Parameter groups: non-mask (base etc.) and mask logits
    mask_params, other_params = [], []
    for n, p in masked_model.named_parameters():
        (mask_params if "mask_logits" in n else other_params).append(p)

    opt = torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr, "weight_decay": 0.01},
            {
                "params": mask_params,
                "lr": mp_cfg.lr,
                "weight_decay": mp_cfg.weight_decay,
            },
        ]
    )

    # Simple linear scheduler for total steps (optional)
    total_steps = num_epochs * len(train_dl)
    pruner = ProgressivePruner(
        masked_model,
        overlays,
        mp_cfg,
        device,
        total_steps_hint=total_steps,
        base_ref_model=base_model,
    )

    global_step = 0
    pbar = tqdm(total=total_steps, desc="TrainWithPruning", leave=False)
    for ep in range(1, num_epochs + 1):
        for batch in train_dl:
            pruner.on_step(global_step)

            batch = _to_device(batch, device)
            out = _forward_for_hf_like(masked_model, batch)
            loss_main = (
                out.loss if hasattr(out, "loss") and out.loss is not None else 0.0
            )

            # L1 sparsity regularization
            l1_terms = [
                m.regularization_loss()
                for m in masked_model.modules()
                if isinstance(m, LinearWithLearnedMask)
            ]
            l1 = (
                torch.stack(l1_terms).mean().to(device)
                if l1_terms
                else torch.tensor(0.0, device=device)
            )

            loss = loss_main + mp_cfg.lmbda_l1 * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            pbar.update(1)

            if mp_cfg.verbose and (global_step % mp_cfg.log_every == 0):
                with torch.no_grad():
                    # Weight the keep ratio by the number of weights per row (out*in).
                    num, den = 0.0, 0.0
                    for m in masked_model.modules():
                        if isinstance(m, LinearWithLearnedMask):
                            k = m.effective_row_mask().mean().item()
                            w_per_row = float(m.base.in_features)
                            rows = float(m.base.out_features)
                            num += k * rows * w_per_row
                            den += rows * w_per_row
                    avg_keep = float(num / max(1.0, den))
                print(
                    f"[progressive] step={global_step} loss={float(loss):.4f} avg_keep≈{avg_keep:.3f}"
                )

    pbar.close()
    if pruner.last_thr is not None:
        for m in overlays.values():
            m.binarize(pruner.last_thr)
    return masked_model, overlays


# ---------------- reporting ----------------


@torch.no_grad()
def calc_sparsity(model: nn.Module) -> Tuple[float, Dict[str, float]]:
    """Compute overall and per-Linear-layer sparsity (% of zeros)."""
    total_zeros, total_elems = 0, 0
    per_layer: Dict[str, float] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            W = m.weight.data
            z = int((W == 0).sum().item())
            total_zeros += z
            total_elems += W.numel()
            per_layer[name] = 100.0 * z / W.numel()
    overall = 100.0 * total_zeros / total_elems if total_elems else 0.0
    return overall, per_layer
