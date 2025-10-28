# cli/prune_eval.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from config import build_default_bert, build_default_vit
from data_utils import is_vision, prepare_dataloaders
from model_utils import get_device
from pipelines.pruning import prune_model
from pipelines.adv_data import build_mixed_pruning_loader, build_adversarial_test_loader
from pipelines.evaluate import evaluate_to_csv


def main():
    ap = argparse.ArgumentParser(
        description="Prune → Evaluate (Original & Adversarial)"
    )
    ap.add_argument("--exp", choices=["bert", "vit"], default="bert")
    ap.add_argument("--model_size", choices=["tiny", "small"], default="small")
    ap.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    ap.add_argument("--methods", nargs="+", default=["wanda", "lbmask"])
    ap.add_argument(
        "--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--outdir", type=str, default="./results_prune_eval")
    # Epsilon configs
    ap.add_argument("--eps-start-nlp", type=float, default=0.0)
    ap.add_argument("--eps-max-nlp", type=float, default=0.25)
    ap.add_argument("--eps-step-nlp", type=float, default=0.01)
    ap.add_argument("--eps-start-vision", type=float, default=0.0)
    ap.add_argument("--eps-max-vision", type=float, default=8 / 255)
    ap.add_argument("--eps-step-vision", type=float, default=2 / 255)
    args = ap.parse_args()

    build = build_default_bert if args.exp == "bert" else build_default_vit
    cfg = build(0, args.methods[0], args.model_size, args.sparsity, args.dataset)
    device = get_device(cfg)

    # Load saved fine-tuned model + processor (fail fast if not present)
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        AutoModelForImageClassification,
        AutoImageProcessor,
    )

    if Path(cfg.save_dir).exists():
        if is_vision(cfg):
            model = AutoModelForImageClassification.from_pretrained(cfg.save_dir)
            processor = AutoImageProcessor.from_pretrained(cfg.save_dir, use_fast=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(cfg.save_dir)
            processor = AutoTokenizer.from_pretrained(cfg.save_dir)
    else:
        raise RuntimeError(
            f"Fine-tuned model not found at '{cfg.save_dir}'. Please run training once."
        )

    train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
    vision_stats = None
    if is_vision(cfg) and hasattr(processor, "image_mean"):
        vision_stats = (tuple(processor.image_mean), tuple(processor.image_std))

    eps_nlp = {
        "start": args.eps_start_nlp,
        "max": args.eps_max_nlp,
        "step": args.eps_step_nlp,
    }
    eps_vis = {
        "start": args.eps_start_vision,
        "max": args.eps_max_vision,
        "step": args.eps_step_vision,
    }

    for m in args.methods:
        for r in args.ratios:
            # Wanda uses calibration mix; LB‑Mask uses train mix by design.
            mix_calib = build_mixed_pruning_loader(
                model,
                cfg,
                calib_dl,
                device,
                r,
                eps_nlp=eps_nlp,
                eps_vision=eps_vis,
                vision_stats=vision_stats,
            )
            mix_train = build_mixed_pruning_loader(
                model,
                cfg,
                train_dl,
                device,
                r,
                eps_nlp=eps_nlp,
                eps_vision=eps_vis,
                vision_stats=vision_stats,
            )
            use_loader = mix_train if m == "lbmask" else mix_calib

            pruned, sp = prune_model(
                model,
                method=m,
                loader=use_loader,
                device=device,
                sparsity=args.sparsity,
            )

            # Original evaluation (multiclass)
            out_base = (
                Path(args.outdir)
                / f"{args.exp}_{args.model_size}_{args.dataset}"
                / m
                / f"ratio_{int(r*100):03d}"
            )
            acc_orig, _ = evaluate_to_csv(
                pruned,
                test_dl,
                cfg,
                device,
                out_csv=out_base / "multiclass_original.csv",
                binary_target_idx=None,
            )

            # Adversarial-only evaluation (multiclass)
            adv_test_dl = build_adversarial_test_loader(
                pruned,
                test_dl,
                cfg,
                device,
                eps_nlp=eps_nlp,
                eps_vision=eps_vis,
                vision_stats=vision_stats,
            )
            acc_adv, _ = evaluate_to_csv(
                pruned,
                adv_test_dl,
                cfg,
                device,
                out_csv=out_base / "multiclass_adversarial.csv",
                binary_target_idx=None,
            )
            print(
                f"[{m} | adv={r:.2f}] sparsity={sp:.1f}% | orig={acc_orig:.4f} | adv={acc_adv:.4f}"
            )


if __name__ == "__main__":
    main()
