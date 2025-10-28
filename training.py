# training.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, default_data_collator
from config import ExpConfig

_WANDB_AVAILABLE = False
try:
    import wandb  # noqa: F401

    _WANDB_AVAILABLE = True
except ImportError:
    pass


def train(model: nn.Module, train_dl: DataLoader, cfg: ExpConfig, device: torch.device):
    """Fine-tune using Hugging Face Trainer."""
    model.to(device)

    # Derive datasets from the existing DataLoader(s).
    train_ds = train_dl.dataset
    eval_ds = None

    steps_per_epoch = max(1, len(train_dl))

    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.save_dir, "trainer_tmp"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.scheduler,
        logging_steps=cfg.logging_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        save_strategy=cfg.save_strategy,  # <- 'no'|'steps'|'epoch'|'best'
        eval_strategy=cfg.eval_strategy,
        dataloader_pin_memory=False,
        seed=cfg.seed,
        fp16=(cfg.fp16 and torch.cuda.is_available()),  # <- gated by --fp16
        report_to=(
            ["wandb"] if (getattr(cfg, "use_wandb", False) and _WANDB_AVAILABLE) else []
        ),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dl.dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
        # tokenizer is optional; we save tokenizer/processor explicitly elsewhere
    )

    # Run training (handles optimizer/scheduler/gradient steps internally)
    trainer.train()
