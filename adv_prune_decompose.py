# adv_prune_decompose.py
# -----------------------------------------------------------------------------
# Orchestrates:
#  - (A) Decomposition/Pruning with only Original set
#  - (B) Decomposition/Pruning with mixed (Original + Adversarial) at various ratios
#  - Evaluation on Original test set (performance) and Adversarial test set (robustness)
#
# Uses:
#  - WANDA one-shot pruning (no fine-tune of weights)
#  - LB-Mask pruning with base weights frozen
#  - Two-head decomposition to evaluate a chosen binary target (optional)
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from config import ExpConfig, build_default_bert, build_default_vit
from data_utils import is_vision, prepare_dataloaders
from model_utils import get_device
from pipelines.adv_data import build_mixed_pruning_loader, build_adversarial_test_loader
from pipelines.pruning import prune_model
from pipelines.decompose import decompose_binary
from pipelines.evaluate import evaluate_to_csv


# ---------- Core runners ----------


def run_decompose_prune_once(
    cfg: ExpConfig,
    *,
    methods: List[str],
    adv_ratio: float,
    outdir: str,
    exp: str,
    model_size: str,
    dataset: str,
    sparsity: float = 0.5,
    target_class: Optional[int] = None,
    eps_start_nlp: float = 0.0,
    eps_max_nlp: float = 0.25,
    eps_step_nlp: float = 0.01,
    eps_start_vision: float = 0.0,
    eps_max_vision: float = 8 / 255,
    eps_step_vision: float = 2 / 255,
):
    """
    Execute one setting (adv_ratio, sparsity) for all methods.
    Prints:
      - Sparsity and accuracy on Original test set
      - Robust accuracy on Adversarial test set
      - (Optional) Binary accuracy of the decomposed [target, negative] head
    """
    device = get_device(cfg)
    if os.path.exists(cfg.save_dir):
        if is_vision(cfg):
            # Load fine-tuned vision model + processor from disk
            from transformers import AutoModelForImageClassification, AutoImageProcessor

            model = AutoModelForImageClassification.from_pretrained(cfg.save_dir)
            processor = AutoImageProcessor.from_pretrained(cfg.save_dir, use_fast=True)
        else:
            # Load fine-tuned NLP model + tokenizer from disk
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model = AutoModelForSequenceClassification.from_pretrained(cfg.save_dir)
            processor = AutoTokenizer.from_pretrained(cfg.save_dir)
    else:
        # Fail fast to prevent evaluating a random-initialized classifier head
        raise RuntimeError(
            f"Fine-tuned model not found at '{cfg.save_dir}'. "
            "Run main.py (or pruning_sweep.py) once to create and save it."
        )

    train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)

    # Collect vision normalization stats if applicable
    vision_stats = None
    if is_vision(cfg):
        # AutoImageProcessor exposes image_mean / image_std used in transforms.Normalize
        mean = tuple(processor.image_mean)  # type: ignore[attr-defined]
        std = tuple(processor.image_std)  # type: ignore[attr-defined]
        vision_stats = (mean, std)

    # Build pruning input loader according to adv_ratio
    eps_nlp = {"start": eps_start_nlp, "max": eps_max_nlp, "step": eps_step_nlp}
    eps_vis = {
        "start": eps_start_vision,
        "max": eps_max_vision,
        "step": eps_step_vision,
    }

    mix_loader_calib = build_mixed_pruning_loader(
        model,
        cfg,
        calib_dl,
        device,
        adv_ratio,
        eps_nlp=eps_nlp,
        eps_vision=eps_vis,
        vision_stats=vision_stats,
    )
    mix_loader_train = build_mixed_pruning_loader(
        model,
        cfg,
        train_dl,
        device,
        adv_ratio,
        eps_nlp=eps_nlp,
        eps_vision=eps_vis,
        vision_stats=vision_stats,
    )

    results = []
    all_csv_rows = []

    for method in methods:
        # LB-Mask -> use full-train mix; Wanda -> use calibration mix
        use_loader = mix_loader_train if method == "lbmask" else mix_loader_calib

        pruned, overall_sparsity = prune_model(
            model,
            method=method,
            loader=use_loader,
            device=device,
            sparsity=sparsity,
        )

        # Original (multiclass) evaluation → CSV
        out_base = (
            Path(outdir)
            / f"{exp}_{model_size}_{dataset}"
            / method
            / f"ratio_{int(adv_ratio*100):03d}"
        )
        acc_orig, _ = evaluate_to_csv(
            pruned,
            test_dl,
            cfg,
            device,
            out_csv=out_base / "multiclass_original.csv",
            binary_target_idx=None,
        )

        # Adversarial test loader & accuracy
        adv_test_loader = build_adversarial_test_loader(
            pruned,
            test_dl,
            cfg,
            device,
            eps_nlp=eps_nlp,
            eps_vision=eps_vis,
            vision_stats=vision_stats,
        )
        robust_acc, _ = evaluate_to_csv(
            pruned,
            adv_test_loader,
            cfg,
            device,
            out_csv=out_base / "multiclass_adversarial.csv",
            binary_target_idx=None,
        )

        row = {
            "method": method,
            "adv_ratio": adv_ratio,
            "sparsity_target": sparsity,
            "sparsity_achieved_pct": overall_sparsity,
            "acc_original": acc_orig,
            "acc_adversarial": robust_acc,
        }

        if target_class is not None:
            decomp = decompose_binary(pruned, target_class, device)
            out_bin = out_base / f"target_{int(target_class):02d}"
            # Original (binary)
            bin_acc_orig, _ = evaluate_to_csv(
                decomp,
                test_dl,
                cfg,
                device,
                out_csv=out_bin / "binary_original.csv",
                binary_target_idx=target_class,
            )
            # Adversarial-only (binary)
            adv_test_loader_bin = build_adversarial_test_loader(
                decomp,
                test_dl,
                cfg,
                device,
                eps_nlp=eps_nlp,
                eps_vision=eps_vis,
                vision_stats=vision_stats,
                binary_target_idx=target_class,
            )
            bin_acc_adv, _ = evaluate_to_csv(
                decomp,
                adv_test_loader_bin,
                cfg,
                device,
                out_csv=out_bin / "binary_adversarial.csv",
                binary_target_idx=target_class,
            )

            row.update(
                {
                    "binary_target": int(target_class),
                    "binary_acc_original": bin_acc_orig,
                    "binary_acc_adversarial": bin_acc_adv,
                }
            )

    return results, all_csv_rows


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial-aware Decomposition & Pruning Runner"
    )
    parser.add_argument("--exp", type=str, default="bert", choices=["bert", "vit"])
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["tiny", "small"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ag_news",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["wanda", "lbmask", "mi"],
        choices=["wanda", "lbmask", "mi"],
    )
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--sparsity", type=float, default=0.5)
    # NLP sweep
    parser.add_argument(
        "--eps-start-nlp",
        type=float,
        default=0.0,
        help="Start epsilon for NLP (embedding space).",
    )
    parser.add_argument(
        "--eps-max-nlp",
        type=float,
        default=0.25,
        help="Max epsilon for NLP (embedding space).",
    )
    parser.add_argument(
        "--eps-step-nlp",
        type=float,
        default=0.01,
        help="Step size for NLP epsilon sweep.",
    )

    # Vision sweep (pixel space; internally converted by mean/std)
    parser.add_argument(
        "--eps-start-vision",
        type=float,
        default=0.0,
        help="Start epsilon in pixel space [0,1].",
    )
    parser.add_argument(
        "--eps-max-vision",
        type=float,
        default=8 / 255,
        help="Max epsilon in pixel space [0,1].",
    )
    parser.add_argument(
        "--eps-step-vision",
        type=float,
        default=2 / 255,
        help="Step size in pixel space [0,1].",
    )

    parser.add_argument(
        "--target_class",
        type=int,
        default=-1,
        help="if >=0, evaluate two-head decomposition",
    )
    parser.add_argument("--outdir", type=str, default="./results_adv_decompose")

    args = parser.parse_args()

    if args.exp == "bert":
        cfg = build_default_bert(
            0, args.methods[0], args.model_size, args.sparsity, args.dataset
        )
    else:
        cfg = build_default_vit(
            0, args.methods[0], args.model_size, args.sparsity, args.dataset
        )

    # Enforce common knobs
    cfg.prune_method = args.methods[0]
    cfg.sparsity_ratio = args.sparsity

    tgt = args.target_class if args.target_class >= 0 else None

    all_rows = []
    all_csv_rows = []
    for r in args.ratios:
        rows, csv_rows = run_decompose_prune_once(
            cfg,
            methods=args.methods,
            adv_ratio=r,
            outdir=args.outdir,
            exp=args.exp,
            model_size=args.model_size,
            dataset=args.dataset,
            sparsity=args.sparsity,
            target_class=tgt,
            eps_start_nlp=args.eps_start_nlp,
            eps_max_nlp=args.eps_max_nlp,
            eps_step_nlp=args.eps_step_nlp,
            eps_start_vision=args.eps_start_vision,
            eps_max_vision=args.eps_max_vision,
            eps_step_vision=args.eps_step_vision,
        )
        all_rows.extend(rows)
        all_csv_rows.extend(csv_rows)

    # Print a compact table at the end
    print("\n==== Summary ====")
    for rr in all_rows:
        msg = (
            f"{rr['method']:>6s} | adv={rr['adv_ratio']:.2f} | "
            f"sp={rr['sparsity_achieved_pct']:.1f}% | "
            f"orig={rr['acc_original']:.4f} | adv={rr['acc_adversarial']:.4f}"
        )
        if "binary_target" in rr:
            msg += (
                f" | bin(orig)={rr['binary_acc_original']:.4f} "
                f"| bin(adv)={rr['binary_acc_adversarial']:.4f}"
            )
        print(msg)


if __name__ == "__main__":
    main()
