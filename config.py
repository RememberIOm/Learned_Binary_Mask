# config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ExpConfig:
    # Generic
    model_name: str
    dataset_name: str
    task_type: str  # "nlp" | "vision"
    num_labels: int

    batch_size: int = 32
    num_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_length: int = 128  # for NLP
    gradient_accumulation_steps: int = 1
    scheduler: str = "linear"  # "linear" | "cosine" | "constant"
    seed: int = 42
    eval_steps: int = 500
    save_strategy: str = "best"  # "no" | "steps" | "epoch" | "best"
    eval_strategy: str = "no"
    logging_steps: int = 50
    fp16: bool = False
    grad_checkpointing: bool = False
    num_workers: int = 4  # for DataLoader

    gpu: Optional[int] = 0
    save_dir: str = "./finetuned_model"
    use_wandb: bool = False
    wandb_project: str = "pruning-experiments"

    # Pruning
    prune_method: str = "lbmask"  # "wanda" | "wanda_nm" | "lbmask" | "mi"
    sparsity_ratio: float = 0.5  # Wanda target per-row fraction (unstructured)
    nm_values: Optional[Tuple[int, int]] = None  # e.g., (2,4) for N:M

    # In-training LB-Mask options
    prune_during_train: bool = True  # True => progressive pruning during training
    prune_schedule: str = "constant"  # constant | linear | cosine
    prune_start_sparsity: float = 0.0
    prune_end_sparsity: float = 0.5
    prune_begin_step: int = 0
    prune_end_step: Optional[int] = None  # if None, will be total_steps
    prune_update_every: int = 100
    hard_apply: bool = False  # permanently zero weights during training
    freeze_base: bool = True  # jointly train base weights when True->False

    # Calibration (Wanda)
    calib_nlp: int = 512
    calib_vision: int = 1024

    # Decomposition
    target_class: int = 0
    apply_decomposition: bool = False  # Whether to apply decomposition or not


def build_default_bert(
    gpu: Optional[int],
    prune_method: str,
    model_size: str,
    sparsity: float,
    dataset: str,
) -> ExpConfig:
    num_labels_map = {"ag_news": 4, "dbpedia_14": 14}
    return ExpConfig(
        model_name=f"prajjwal1/bert-{model_size}",
        dataset_name=dataset,
        task_type="nlp",
        num_labels=num_labels_map.get(dataset, 10),
        batch_size=32,
        num_epochs=3,
        lr=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        max_length=128,
        scheduler="linear",
        seed=42,
        save_strategy="epoch",
        eval_strategy="no",
        logging_steps=50,
        fp16=False,
        gpu=gpu,
        save_dir=f"./finetuned_bert_{model_size}_{dataset}",
        prune_method=prune_method,
        sparsity_ratio=sparsity,
        use_wandb=False,
        num_workers=4,
    )


def build_default_vit(
    gpu: Optional[int],
    prune_method: str,
    model_size: str,
    sparsity: float,
    dataset: str,
) -> ExpConfig:
    # ViT on Fashion MNIST (10-way)
    return ExpConfig(
        model_name=f"facebook/deit-{model_size}-patch16-224",
        dataset_name=dataset,
        task_type="vision",
        num_labels=10,
        batch_size=32,
        num_epochs=5,
        lr=5e-5,
        gpu=gpu,
        save_dir=f"./finetuned_deit_{model_size}_{dataset}",
        prune_method=prune_method,
        sparsity_ratio=sparsity,
        nm_values=(2, 4) if prune_method == "wanda_nm" else None,
        use_wandb=False,
    )
