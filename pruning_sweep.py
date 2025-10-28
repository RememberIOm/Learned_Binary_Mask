"""
Pruning Sweep Runner (WANDA vs LB-Mask) + Plotter

- Sweeps sparsity in [0.1, 0.9] (step=0.1)
- Compares WANDA (unstructured) and LB-Mask
- Works for both NLP (BERT on AG News, Yahoo) and Vision (DeiT on Fashion-MNIST, CIFAR-10)
- Saves per-sweep metrics to CSV and generates accuracy-vs-sparsity plots

Usage
-----
# Default sweep (two pairs):
python pruning_sweep.py

# Specify experiments and options:
python pruning_sweep.py \
  --experiments "bert:small:ag_news" "vit:small:fashion_mnist" \
  --methods wanda lbmask \
  --sparsities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --outdir ./results_sweep

The saved artifacts include:
- CSV: outdir/{exp_name}_{method}.csv
- PNG: outdir/plots/{exp_name}_accuracy_vs_sparsity.png
"""

import os
import csv
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Local modules
from config import ExpConfig, build_default_bert, build_default_vit
from data_utils import is_vision, prepare_dataloaders
from evaluation import evaluate
from model_utils import get_device, load_model_and_processor
from training import train
from learned_binary_mask_pruning import (
    MaskPruneConfig,
    export_pruned_copy,
    calc_sparsity,  # generic sparsity util
    train_with_progressive_pruning,
)
from wanda_pruning import wanda_prune
from decomposition import decompose_to_target_vs_rest  # two-head [target, negative]


# ------------------------- CSV helpers (FIX) -------------------------


def _gather_fieldnames(rows):
    """Compute a stable union of keys across rows for CSV header."""
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    return keys


def _write_csv(path: Path, rows):
    """Write rows (list of dict) to CSV, handling missing/extra keys safely."""
    if not rows:
        return
    fieldnames = _gather_fieldnames(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row_out)


# ------------------------- Sweep logic -------------------------


def ensure_base_model(cfg: ExpConfig, device: torch.device):
    """
    Make sure a fine-tuned base model exists on disk.
    If not found at cfg.save_dir, we'll train from scratch and save there.
    """
    save_dir = Path(cfg.save_dir)
    if save_dir.exists():
        # Load from saved directory
        if is_vision(cfg):
            from transformers import AutoModelForImageClassification, AutoImageProcessor

            model = AutoModelForImageClassification.from_pretrained(cfg.save_dir)
            processor = AutoImageProcessor.from_pretrained(cfg.save_dir, use_fast=True)
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model = AutoModelForSequenceClassification.from_pretrained(cfg.save_dir)
            processor = AutoTokenizer.from_pretrained(cfg.save_dir)
        train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
        return model, processor, train_dl, test_dl, calib_dl

    # No saved model => train
    model, processor = load_model_and_processor(cfg)
    train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
    train(model, train_dl, cfg, device)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.save_dir)
    if is_vision(cfg):
        processor.save_pretrained(cfg.save_dir, safe_serialization=True)
    else:
        processor.save_pretrained(cfg.save_dir)
    # Rebuild loaders after save (consistent with main.py pattern)
    train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
    return model, processor, train_dl, test_dl, calib_dl


def run_wanda_once(
    model,
    calib_dl,
    test_dl,
    cfg: ExpConfig,
    device: torch.device,
    sparsity: float,
    target_class: int,
    apply_decomposition: bool,
) -> Dict:
    """
    Run WANDA pruning at a given sparsity; return metrics.
    """
    pruned = wanda_prune(
        model,
        calib_dl,
        sparsity_ratio=sparsity,
        device=device,
        method="wanda",
        nm_values=None,
    )
    overall_sparsity, _ = calc_sparsity(pruned)
    acc_mc, _ = evaluate(pruned, test_dl, cfg, device)

    row = {
        "method": "wanda",
        "target_sparsity": sparsity,
        "achieved_sparsity": overall_sparsity / 100.0,
        "accuracy": acc_mc,
    }

    if apply_decomposition:
        # Replace final head with [target, negative] on the PRUNED COPY only
        decomp = decompose_to_target_vs_rest(pruned, target_class).to(device)
        acc_bin, _ = evaluate(
            decomp, test_dl, cfg, device, binary_target_idx=target_class
        )  # requires main.py change
        row.update({"binary_accuracy": acc_bin, "binary_target_class": target_class})

    return row


def run_sweep_for_experiment(
    exp: str,
    methods: List[str],
    sparsities: List[float],
    outdir: Path,
    gpu: Optional[int] = 0,
    lbmask_schedules: Optional[list] = None,
    *,
    target_class: int = 0,
    apply_decomposition: bool = False,
) -> List[Dict]:
    """
    Run a sparsity sweep for one experiment descriptor.

    exp format: "bert:small:ag_news" or "vit:small:fashion_mnist"
    methods: e.g., ["wanda", "lbmask"]
    sparsities: list of floats in (0,1)
    """
    kind, size, dataset = exp.split(":")
    build_fn = build_default_bert if kind == "bert" else build_default_vit

    results_dir = outdir
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    for method in methods:
        # Build base config for this (model,dataset,method)
        cfg = build_fn(
            gpu,
            method,
            size,
            sparsities[0],  # placeholder; we will override per-run
            dataset,
        )
        device = get_device(cfg)

        # Ensure base fine-tuned model exists and get loaders
        model, processor, train_dl, test_dl, calib_dl = ensure_base_model(cfg, device)

        # Evaluate baseline (unpruned) once (same for both methods)
        base_acc, _ = evaluate(model, test_dl, cfg, device)

        if method == "lbmask":
            # Train with schedule for each (schedule, sparsity) pair.
            for sched in lbmask_schedules or ["constant", "linear", "cosine"]:
                for s in sparsities:
                    mp_cfg = MaskPruneConfig(
                        granularity="out",
                        lr=5e-3,
                        lmbda_l1=1e-3,
                        num_epochs=cfg.num_epochs,
                        skip_final_classifier=True,
                        prune_during_train=True,
                        schedule=sched,  # "constant" | "linear" | "cosine"
                        start_sparsity=0.0,
                        end_sparsity=s,
                        begin_step=0,
                        end_step=cfg.num_epochs * len(train_dl),
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
                    pruned_model = export_pruned_copy(model, overlays).to(device)
                    overall_sparsity, _ = calc_sparsity(pruned_model)
                    acc_mc, _ = evaluate(pruned_model, test_dl, cfg, device)
                    row = {
                        "method": f"lbmask/{sched}",
                        "target_sparsity": s,
                        "achieved_sparsity": overall_sparsity / 100.0,
                        "accuracy": acc_mc,
                        "exp": exp,
                        "model_name": cfg.model_name,
                        "dataset": cfg.dataset_name,
                        "base_accuracy": base_acc,
                    }

                    if apply_decomposition:
                        decomp = decompose_to_target_vs_rest(
                            pruned_model, target_class
                        ).to(device)
                        acc_bin, _ = evaluate(
                            decomp,
                            test_dl,
                            cfg,
                            device,
                            binary_target_idx=target_class,
                        )
                        row.update(
                            {
                                "binary_accuracy": acc_bin,
                                "binary_target_class": target_class,
                            }
                        )

                    all_rows.append(row)
        elif method == "wanda":
            # WANDA remains one-shot per sparsity (by design).
            for s in sparsities:
                cfg.sparsity_ratio = s
                row = run_wanda_once(
                    model,
                    calib_dl,
                    test_dl,
                    cfg,
                    device,
                    s,
                    target_class,
                    apply_decomposition,
                )
                row.update(
                    {
                        "exp": exp,
                        "model_name": cfg.model_name,
                        "dataset": cfg.dataset_name,
                        "base_accuracy": base_acc,
                    }
                )
                all_rows.append(row)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Save CSV per-method for convenience (handles heterogeneous keys)
        csv_path = results_dir / f"{exp.replace(':','_')}_{method}.csv"
        if method == "lbmask":
            method_rows = [
                rr
                for rr in all_rows
                if rr["method"].startswith("lbmask/") and rr["exp"] == exp
            ]
        else:
            method_rows = [
                rr for rr in all_rows if rr["method"] == method and rr["exp"] == exp
            ]
        _write_csv(csv_path, method_rows)

    # Also save a merged CSV for the experiment
    merged_csv = results_dir / f"{exp.replace(':','_')}_ALL.csv"
    exp_rows = [rr for rr in all_rows if rr["exp"] == exp]
    _write_csv(merged_csv, exp_rows)

    # Plot accuracy vs sparsity (one chart per experiment; multiple lines for methods)
    plt.figure()
    lines = sorted({r["method"] for r in all_rows if r["exp"] == exp})
    for mname in lines:
        xs = [
            r["target_sparsity"]
            for r in all_rows
            if r["method"] == mname and r["exp"] == exp
        ]
        ys = [
            r["accuracy"] for r in all_rows if r["method"] == mname and r["exp"] == exp
        ]
        plt.plot(xs, ys, marker="o", label=mname.upper())
    plt.xlabel("Target sparsity (ratio)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Sparsity — {exp}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    fig_path = plots_dir / f"{exp.replace(':','_')}_accuracy_vs_sparsity.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Run pruning sweeps (WANDA vs LB-Mask) and plot results."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["bert:small:ag_news", "vit:small:fashion_mnist"],
        help="List like 'bert:small:ag_news' or 'vit:small:fashion_mnist'",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wanda", "lbmask"],
        help="Pruning methods to compare",
    )
    parser.add_argument(
        "--sparsities",
        nargs="+",
        type=float,
        default=[round(x, 1) for x in np.linspace(0.1, 0.9, 9)],
        help="Target sparsity ratios (e.g., 0.1 0.2 ... 0.9)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU id (use -1 to auto-select or None)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./results_pruning_sweep",
        help="Directory to save CSVs and plots",
    )
    parser.add_argument(
        "--lbmask_schedules",
        nargs="+",
        default=["constant", "linear", "cosine"],
        choices=["constant", "linear", "cosine"],
        help="Schedules to compare when --lbmask_progressive is set.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="index treated as 'target' for two-head decomposition",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="after pruning, replace final head with [target, negative] and eval binary",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Make runs reproducible-ish
    torch.manual_seed(42)
    np.random.seed(42)

    all_exps_rows: List[Dict] = []
    for exp in args.experiments:
        rows = run_sweep_for_experiment(
            exp,
            args.methods,
            args.sparsities,
            outdir,
            gpu=args.gpu,
            lbmask_schedules=args.lbmask_schedules,
            target_class=args.target_class,
            apply_decomposition=args.decompose,
        )
        all_exps_rows.extend(rows)

    # Save a grand-merged CSV as well
    if all_exps_rows:
        merged = outdir / "ALL_experiments.csv"
        _write_csv(merged, all_exps_rows)

    print(f"✅ Done. Results saved under: {outdir.resolve()}")
    print("   - Per-experiment CSVs and ALL_experiments.csv")
    print("   - Plots in the 'plots' subfolder")


if __name__ == "__main__":
    main()
