# main.py
# ---------------------------------------------------------------------
# Unified experiment runner for two pruning methods:
# - "wanda" / "wanda_nm": One-shot Wanda pruning (with optional N:M)
# - "lbmask": Learnable Binary-Mask pruning with line-search projection
# Internal comments are kept in English.
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from datasets import load_dataset
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from tqdm.auto import tqdm
import warnings

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Local modules
from learned_binary_mask_pruning import (
    MaskPruneConfig,
    build_masked_model_from,
    train_masks,
    project_by_line_search,
    export_pruned_copy,
    calc_sparsity,  # generic sparsity util
    train_with_progressive_pruning,
)
from wanda_pruning import wanda_prune
from decomposition import decompose_to_target_vs_rest
from adv_utils import map_target_vs_rest

# ------------------------- Optional logging -------------------------
try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# =========================== Config ===========================


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
    save_strategy: str = "epoch"  # "epoch" | "steps"
    fp16: bool = False
    grad_checkpointing: bool = False

    gpu: Optional[int] = 0
    save_dir: str = "./finetuned_model"
    use_wandb: bool = False
    wandb_project: str = "pruning-experiments"

    # Pruning
    prune_method: str = "lbmask"  # "wanda" | "wanda_nm" | "lbmask"
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


# =========================== Utilities ===========================


def get_device(cfg: ExpConfig) -> torch.device:
    """Pick a device based on availability and cfg.gpu."""
    if torch.cuda.is_available():
        if cfg.gpu is None or int(cfg.gpu) < 0:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(cfg.gpu)}")
    return torch.device("cpu")


def class_names_for(dataset_name: str) -> Optional[List[str]]:
    """Return human-friendly class names when available."""
    if dataset_name == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    if dataset_name == "fashion_mnist":
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    if dataset_name == "ag_news":
        return ["World", "Sports", "Business", "Sci/Tech"]
    if dataset_name == "dbpedia_14":
        return [
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ]
    return None


def is_vision(cfg: ExpConfig) -> bool:
    return cfg.task_type == "vision"


# ====================== Model & Data Loading ======================


def load_model_and_processor(cfg: ExpConfig):
    """
    Load HF model + pre/post processor based on task type.
    For ViT, we replace the classifier head to match num_labels.
    """
    if cfg.task_type == "nlp":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=cfg.num_labels
        )
        return model, tokenizer

    # Vision
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, cfg.num_labels)
    return model, image_processor


def prepare_dataloaders(cfg: ExpConfig, processor):
    """
    Build train/test/calibration DataLoaders.
    Output batches are dicts with unified keys:
      - NLP: {input_ids, attention_mask, labels}
      - Vision: {pixel_values, labels}
    """
    if cfg.task_type == "nlp":
        if cfg.dataset_name == "ag_news":
            ds = load_dataset(cfg.dataset_name)
        elif cfg.dataset_name == "dbpedia_14":
            ds = load_dataset("dbpedia_14")

            # DBpedia: combine title + content → text
            def _build_text(ex):
                title = ex.get("title", "") or ""
                content = ex.get("content", "") or ""
                ex["text"] = " ".join([s for s in (title, content) if s]).strip()
                return ex

            ds = ds.map(_build_text)
        else:
            raise ValueError(f"Unknown NLP dataset: {cfg.dataset_name}")

        def tok_fn(ex):
            # Tokenize prepared 'text'
            return processor(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length,
            )

        tokenized = ds.map(tok_fn, batched=True)

        # Drop raw text-ish columns (keep only model inputs + labels)
        drop_cols = {
            "text",
            "question_title",
            "question_content",
            "best_answer",
            "title",
            "content",
            "id",
            "idx",
            "guid",
        }
        keep = [c for c in tokenized["train"].column_names if c not in drop_cols]
        tokenized = tokenized.select_columns(keep)

        # Normalize label column name
        train_cols = set(tokenized["train"].column_names)
        if "labels" in train_cols:
            pass
        elif "label" in train_cols:
            tokenized = tokenized.rename_column("label", "labels")
        elif "topic" in train_cols:
            tokenized = tokenized.rename_column("topic", "labels")
        else:
            raise ValueError(
                f"Could not find label column. Available: {sorted(train_cols)}"
            )

        tokenized.set_format("torch")

        train_ds = tokenized["train"]
        test_dl = DataLoader(tokenized["test"], batch_size=cfg.batch_size)

        calib_idx = np.random.choice(
            len(train_ds), min(cfg.calib_nlp, len(train_ds)), replace=False
        )
        calib_ds = Subset(train_ds, calib_idx)

        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        calib_dl = DataLoader(calib_ds, batch_size=8)
        return train_dl, test_dl, calib_dl

    # Vision
    size_cfg = processor.size
    tgt = (
        size_cfg
        if isinstance(size_cfg, int)
        else size_cfg.get("height") or size_cfg.get("shortest_edge") or 224
    )

    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    def to_3ch(x):
        # If grayscale, repeat to 3 channels
        return x.repeat(3, 1, 1) if x.size(0) == 1 else x

    transform = Compose(
        [
            Resize(tgt),
            ToTensor(),
            to_3ch,
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    if cfg.dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10

        train_raw = CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_raw = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif cfg.dataset_name == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        train_raw = FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_raw = FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown vision dataset: {cfg.dataset_name}")

    # Wrap to dict-style batches for a unified forward
    class VisionDict(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __getitem__(self, idx):
            px, y = self.base[idx]
            return {"pixel_values": px, "labels": y}

        def __len__(self):
            return len(self.base)

    train_ds = VisionDict(train_raw)
    test_ds = VisionDict(test_raw)

    calib_idx = np.random.choice(
        len(train_ds), min(cfg.calib_vision, len(train_ds)), replace=False
    )
    calib_ds = Subset(train_ds, calib_idx)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)
    calib_dl = DataLoader(calib_ds, batch_size=8)
    return train_dl, test_dl, calib_dl


# =========================== Training ===========================


def train(model: nn.Module, train_dl: DataLoader, cfg: ExpConfig, device: torch.device):
    """Fine-tune using Hugging Face Trainer."""
    model.to(device)

    # Derive datasets from the existing DataLoader(s).
    # We keep the signature unchanged to avoid touching other call sites.
    train_ds = train_dl.dataset
    eval_ds = None  # Keep evaluation off here; main/runers will evaluate separately.

    # Keep behavior close to the original loop:
    # - no intermediate eval/saving by default
    # - simple linear schedule with warmup
    # - logging every ~10% of an epoch
    steps_per_epoch = max(1, len(train_dl))
    logging_steps = max(10, steps_per_epoch // 10)

    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.save_dir, "trainer_tmp"),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        logging_steps=logging_steps,
        save_strategy="none",  # keep model saving under our control (below)
        evaluation_strategy="no",  # we run eval after training via existing code
        dataloader_pin_memory=False,  # safer across CPU/GPU switching
        seed=42,
        fp16=torch.cuda.is_available(),
        report_to=(
            ["wandb"] if (getattr(cfg, "use_wandb", False) and _WANDB_AVAILABLE) else []
        ),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        # tokenizer is optional; we save tokenizer/processor explicitly elsewhere
    )

    # Run training (handles optimizer/scheduler/gradient steps internally)
    trainer.train()


# =========================== Evaluation ===========================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_dl: DataLoader,
    cfg: ExpConfig,
    device: torch.device,
    binary_target_idx: Optional[int] = None,
) -> Tuple[float, Dict]:
    """Return accuracy and full sklearn classification report dict."""
    model.eval()
    model.to(device)

    preds, labels = [], []
    for batch in tqdm(test_dl, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["labels"]
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        pred = torch.argmax(logits, dim=-1)

        if binary_target_idx is None:
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
        else:
            # Map GT labels to {0:target, 1:negative} to match TwoHeadLinear order.
            y_bin = map_target_vs_rest(y, int(binary_target_idx))
            preds.extend(pred.cpu().numpy())  # 0=target, 1=negative
            labels.extend(y_bin.cpu().numpy())

    if binary_target_idx is None:
        labels_param = list(range(cfg.num_labels))  # e.g., [0,1,2,3]
        names = class_names_for(cfg.dataset_name) or [str(i) for i in labels_param]
    else:
        labels_param = [0, 1]  # 0=target, 1=negative
        names = ["target", "negative"]

    # classification_report with fixed labels and safe zero_division
    rep_dict = classification_report(
        labels,
        preds,
        labels=labels_param,
        target_names=names,
        output_dict=True,
        digits=4,
        zero_division=0,  # avoid warnings/errors for support=0
    )

    print("\n--- Classification Report ---")
    print(
        classification_report(
            labels,
            preds,
            labels=labels_param,
            target_names=names,
            digits=4,
            zero_division=0,
        )
    )
    print("-----------------------------\n")
    return rep_dict["accuracy"], rep_dict


# =========================== Orchestration ===========================


def maybe_init_wandb(cfg: ExpConfig, run_name: str):
    if cfg.use_wandb and _WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb_project, config=vars(cfg), name=run_name)


def run_experiment(cfg: ExpConfig):
    device = get_device(cfg)
    print(f"Device: {device}")

    # Load or train & save
    if os.path.exists(cfg.save_dir):
        print(f"✅ Loading fine-tuned model from '{cfg.save_dir}'")
        if is_vision(cfg):
            model = AutoModelForImageClassification.from_pretrained(cfg.save_dir)
            processor = AutoImageProcessor.from_pretrained(cfg.save_dir, use_fast=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(cfg.save_dir)
            processor = AutoTokenizer.from_pretrained(cfg.save_dir)
        train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
    else:
        print("💾 No saved model found. Starting from scratch.")
        model, processor = load_model_and_processor(cfg)
        train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
        maybe_init_wandb(
            cfg, f"{cfg.model_name.split('/')[-1]}-{cfg.dataset_name}-train"
        )
        train(model, train_dl, cfg, device)
        os.makedirs(cfg.save_dir, exist_ok=True)
        model.save_pretrained(cfg.save_dir)
        if is_vision(cfg):
            processor.save_pretrained(cfg.save_dir, safe_serialization=True)
        else:
            processor.save_pretrained(cfg.save_dir)
        if cfg.use_wandb and _WANDB_AVAILABLE:
            wandb.finish()

    # Always (re)build loaders to ensure availability after load
    if "train_dl" not in locals():
        train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)

    # Evaluate before pruning
    maybe_init_wandb(
        cfg, f"{cfg.model_name.split('/')[-1]}-{cfg.dataset_name}-eval_before"
    )
    base_sparsity, _ = calc_sparsity(model)
    acc0, _ = evaluate(model, test_dl, cfg, device)
    print(f"Sparsity (before): {base_sparsity:.2f}%  |  Accuracy: {acc0:.4f}")
    if cfg.use_wandb and _WANDB_AVAILABLE:
        wandb.log({"original_accuracy": acc0, "original_sparsity": base_sparsity})
        wandb.finish()

    # Apply pruning method
    if cfg.prune_method in ("wanda", "wanda_nm"):
        pruned = wanda_prune(
            model,
            calib_dl,
            sparsity_ratio=cfg.sparsity_ratio,
            device=device,
            method=cfg.prune_method,
            nm_values=cfg.nm_values,
        )
        pruned_sparsity, layer_sparsity = calc_sparsity(pruned)
        acc1, _ = evaluate(pruned, test_dl, cfg, device)
        print(
            f"Sparsity (after target {cfg.sparsity_ratio*100:.0f}%): {pruned_sparsity:.2f}%"
        )
        print(f"Accuracy (after): {acc1:.4f}")
        if cfg.use_wandb and _WANDB_AVAILABLE:
            wandb.log(
                {
                    "pruned_accuracy": acc1,
                    "pruned_sparsity": pruned_sparsity,
                    "pruned_layers_sparsity": layer_sparsity,
                }
            )
            wandb.finish()

        if cfg.apply_decomposition:
            decomposed = decompose_to_target_vs_rest(pruned, cfg.target_class).to(
                device
            )
            acc_bin, _ = evaluate(
                decomposed, test_dl, cfg, device, binary_target_idx=cfg.target_class
            )
            print(
                f"[decomp] Binary accuracy (target={cfg.target_class}): {acc_bin:.4f}"
            )

    elif cfg.prune_method == "lbmask":
        mp_cfg = MaskPruneConfig(
            granularity="out",
            # Mask learning hyperparams
            lr=5e-3,
            lmbda_l1=1e-3,
            num_epochs=cfg.num_epochs,
            skip_final_classifier=True,
            # Progressive schedule
            freeze_base=cfg.freeze_base,
            prune_during_train=True,
            schedule=cfg.prune_schedule,  # "constant" | "linear" | "cosine"
            start_sparsity=cfg.prune_start_sparsity,
            end_sparsity=cfg.prune_end_sparsity,
            begin_step=cfg.prune_begin_step,
            end_step=(
                cfg.prune_end_step
                if cfg.prune_end_step is not None
                else (cfg.num_epochs * len(train_dl))
            ),
            update_every=cfg.prune_update_every,
            hard_apply=cfg.hard_apply,
            zero_bias_when_masked=True,
        )

        masked_model, overlays = train_with_progressive_pruning(
            base_model=model,
            train_dl=train_dl,
            mp_cfg=mp_cfg,
            base_lr=cfg.lr,
            num_epochs=cfg.num_epochs,
            device=device,
        )

        # Export a clean pruned copy (no wrappers) using the last thresholds
        pruned_model = export_pruned_copy(model, overlays).to(device)
        overall_sparsity, _ = calc_sparsity(pruned_model)
        acc1, _ = evaluate(pruned_model, test_dl, cfg, device)

        print(f"[progressive] Acc={acc1:.4f} | Sparsity={overall_sparsity:.2f}%")

        if cfg.use_wandb and _WANDB_AVAILABLE:
            wandb.log({"pruned_accuracy": acc1, "pruned_sparsity": overall_sparsity})
            wandb.finish()

        if cfg.apply_decomposition:
            decomposed = decompose_to_target_vs_rest(pruned_model, cfg.target_class).to(
                device
            )
            acc_bin, _ = evaluate(
                decomposed, test_dl, cfg, device, binary_target_idx=cfg.target_class
            )
            print(
                f"[decomp] Binary accuracy (target={cfg.target_class}): {acc_bin:.4f}"
            )
    else:
        raise ValueError(f"Unknown prune_method: {cfg.prune_method}")


# =========================== CLI presets ===========================


def build_default_bert(args, gpu) -> ExpConfig:
    num_labels_map = {
        "ag_news": 4,
        "dbpedia_14": 14,
    }

    return ExpConfig(
        model_name=f"prajjwal1/bert-{args.model_size}",
        dataset_name=args.dataset,
        task_type="nlp",
        num_labels=num_labels_map.get(args.dataset, 10),
        # ---- Bind CLI hyperparameters ----
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum_steps,
        scheduler=args.scheduler,
        seed=args.seed,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        grad_checkpointing=args.grad_checkpointing,
        gpu=gpu,
        save_dir=f"./finetuned_bert_{args.model_size}_{args.dataset}",
        prune_method=args.method,
        sparsity_ratio=args.sparsity,
        use_wandb=False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pruning Experiments (Wanda / Learned Binary Mask)"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (use -1 for auto)")
    parser.add_argument(
        "--exp",
        type=str,
        default="bert",
        choices=["bert", "vit"],
        help="which preset to run",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small"],
        help="model size",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ag_news",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        help="dataset name",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lbmask",
        choices=["wanda", "wanda_nm", "lbmask"],
        help="pruning method",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="target sparsity ratio (e.g., 0.5)"
    )
    parser.add_argument(
        "--prune_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument("--prune_start", type=float, default=0.0)
    parser.add_argument("--prune_end", type=float, default=0.8)
    parser.add_argument("--prune_begin_step", type=int, default=0)
    parser.add_argument("--prune_end_step", type=int, default=-1)
    parser.add_argument("--prune_update_every", type=int, default=100)
    parser.add_argument("--hard_apply", action="store_true")
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="index of the class to serve as the 'target'",
    )
    parser.add_argument(
        "--no_decomposition",
        action="store_true",
        help="skip two-head decomposition (keep multiclass head)",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--save_strategy", type=str, default="epoch", choices=["epoch", "steps"]
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")

    args = parser.parse_args()

    cfg = (
        build_default_bert(args, None if args.gpu < 0 else args.gpu)
        if args.exp == "bert"
        else build_default_vit(
            args.gpu, args.method, args.model_size, args.sparsity, args.dataset
        )
    )

    # ---- CLI -> ExpConfig overrides (names + minor normalization) ----
    # Keep method selected on CLI (already used by presets, but set explicitly for clarity)
    cfg.prune_method = args.method

    # Map CLI schedule knobs into ExpConfig field names
    cfg.prune_schedule = args.prune_schedule
    cfg.prune_start_sparsity = args.prune_start  # CLI uses --prune_start
    cfg.prune_end_sparsity = args.prune_end  # CLI uses --prune_end
    cfg.prune_begin_step = args.prune_begin_step
    # Normalize -1 to None (means "use total_steps")
    cfg.prune_end_step = None if args.prune_end_step < 0 else args.prune_end_step
    cfg.prune_update_every = args.prune_update_every
    cfg.hard_apply = args.hard_apply

    cfg.target_class = args.target_class
    cfg.apply_decomposition = not args.no_decomposition

    run_experiment(cfg)
