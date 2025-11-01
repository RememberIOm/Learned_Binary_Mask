# core/bootstrap_training.py
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn, optim
from core.model_utils import load_model_and_processor
from core.data_utils import prepare_dataloaders
from core.data_utils import is_vision


def train_initial_model(
    cfg,
    device,
    epochs=3,
    lr=5e-5,
    freeze_backbone=True,
    outdir="./models/checkpoints_bootstrap",
) -> Path:
    """
    Train an initial task model BEFORE pruning. Do NOT call this after pruning.
    Notes:
        - If `freeze_backbone=True`, only classification head is trained (recommended).
        - This function exists solely to bootstrap a task model when no finetuned checkpoint is available.
    """
    model, processor = load_model_and_processor(cfg)
    train_dl, val_dl, _ = prepare_dataloaders(cfg, processor)
    model.to(device)

    if freeze_backbone:
        for n, p in model.named_parameters():
            if "classifier" not in n and "fc" not in n and "head" not in n:
                p.requires_grad = False

    # Simple optimizer/scheduler (robust defaults)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            logits = model(**inputs).logits
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # (optional) quick val
        model.eval()
        with torch.no_grad():
            _ = sum(
                (
                    model(
                        **{k: v.to(device) for k, v in b.items() if k != "labels"}
                    ).logits.argmax(1)
                    == b["labels"].to(device)
                )
                .float()
                .sum()
                .item()
                for b in val_dl
            )
        model.train()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = outdir / f"{cfg.exp}_{cfg.model_size}_{cfg.dataset}_bootstrap.pt"
    torch.save(model.state_dict(), ckpt)
    return ckpt
