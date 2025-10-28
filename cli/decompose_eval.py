# cli/decompose_eval.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from config import build_default_bert, build_default_vit
from data_utils import is_vision, prepare_dataloaders
from model_utils import get_device
from pipelines.decompose import decompose_binary
from pipelines.adv_data import build_adversarial_test_loader
from pipelines.evaluate import evaluate_to_csv


def main():
    ap = argparse.ArgumentParser(
        description="Decompose → Evaluate (Original & Adversarial)"
    )
    ap.add_argument("--exp", choices=["bert", "vit"], default="bert")
    ap.add_argument("--model_size", choices=["tiny", "small"], default="small")
    ap.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    ap.add_argument(
        "--targets", nargs="+", type=int, default=[0]
    )  # supports multiple target classes
    ap.add_argument("--outdir", type=str, default="./results_decompose_eval")
    ap.add_argument("--eps-start-nlp", type=float, default=0.0)
    ap.add_argument("--eps-max-nlp", type=float, default=0.25)
    ap.add_argument("--eps-step-nlp", type=float, default=0.01)
    ap.add_argument("--eps-start-vision", type=float, default=0.0)
    ap.add_argument("--eps-max-vision", type=float, default=8 / 255)
    ap.add_argument("--eps-step-vision", type=float, default=2 / 255)
    args = ap.parse_args()

    build = build_default_bert if args.exp == "bert" else build_default_vit
    # method/sparsity params are irrelevant for pure decomposition eval
    cfg = build(0, "wanda", args.model_size, 0.0, args.dataset)
    device = get_device(cfg)

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
        raise RuntimeError(f"Fine-tuned model not found at '{cfg.save_dir}'.")

    train_dl, test_dl, _ = prepare_dataloaders(cfg, processor)
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

    for t in args.targets:
        decomp = decompose_binary(model, t, device)
        base = (
            Path(args.outdir)
            / f"{args.exp}_{args.model_size}_{args.dataset}"
            / f"target_{t:02d}"
        )

        # Original (binary)
        acc_o, _ = evaluate_to_csv(
            decomp,
            test_dl,
            cfg,
            device,
            out_csv=base / "binary_original.csv",
            binary_target_idx=t,
        )

        # Adversarial-only (binary)
        adv_test_dl = build_adversarial_test_loader(
            decomp,
            test_dl,
            cfg,
            device,
            eps_nlp=eps_nlp,
            eps_vision=eps_vis,
            vision_stats=vision_stats,
            binary_target_idx=t,
        )
        acc_a, _ = evaluate_to_csv(
            decomp,
            adv_test_dl,
            cfg,
            device,
            out_csv=base / "binary_adversarial.csv",
            binary_target_idx=t,
        )
        print(f"[target={t}] orig={acc_o:.4f} | adv={acc_a:.4f}")


if __name__ == "__main__":
    main()
