# main.py

from __future__ import annotations

import argparse
import os
import warnings

# English comment: Optional logging setup
try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

from sklearn.exceptions import UndefinedMetricWarning

# English comment: Import all functionalities from the new modules.
from config import ExpConfig, build_default_bert, build_default_vit
from data_utils import is_vision, prepare_dataloaders
from evaluation import evaluate
from model_utils import get_device, load_model_and_processor
from training import train
from learned_binary_mask_pruning import (
    MaskPruneConfig,
    export_pruned_copy,
    calc_sparsity,
    train_with_progressive_pruning,
)
from wanda_pruning import wanda_prune
from decomposition import decompose_to_target_vs_rest

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
        model, processor = load_model_and_processor(cfg)
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
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="best",
        choices=["no", "steps", "epoch", "best"],
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="no",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.exp == "bert":
        cfg = build_default_bert(
            args.gpu, args.method, args.model_size, args.sparsity, args.dataset
        )
    else:
        cfg = build_default_vit(
            args.gpu, args.method, args.model_size, args.sparsity, args.dataset
        )

    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.warmup_ratio = args.warmup_ratio
    cfg.max_length = args.max_length
    cfg.scheduler = args.scheduler
    cfg.seed = args.seed
    cfg.fp16 = args.fp16
    cfg.save_strategy = args.save_strategy
    cfg.eval_strategy = args.eval_strategy
    cfg.logging_steps = args.logging_steps
    cfg.num_workers = args.num_workers

    # pruning-specific
    cfg.prune_schedule = args.prune_schedule
    cfg.prune_start_sparsity = args.prune_start
    cfg.prune_end_sparsity = args.prune_end
    cfg.prune_begin_step = args.prune_begin_step
    cfg.prune_end_step = None if args.prune_end_step < 0 else args.prune_end_step
    cfg.prune_update_every = args.prune_update_every
    cfg.hard_apply = args.hard_apply

    # decomposition options
    cfg.target_class = args.target_class
    cfg.apply_decomposition = not args.no_decomposition

    run_experiment(cfg)
