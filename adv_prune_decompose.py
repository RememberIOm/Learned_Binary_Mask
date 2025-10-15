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
import csv
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from main import (
    ExpConfig,
    get_device,
    is_vision,
    load_model_and_processor,
    prepare_dataloaders,
    evaluate,  # multiclass evaluation on original test set
)
from wanda_pruning import wanda_prune
from learned_binary_mask_pruning import (
    MaskPruneConfig,
    train_with_progressive_pruning,
    export_pruned_copy,
    calc_sparsity,
)
from decomposition import decompose_to_target_vs_rest
from adv_utils import (
    nlp_to_embeds_dataset,
    nlp_adv_dataset,
    vision_adv_dataset,
    make_mixed_loader,
    DictListDataset,
)

# ---------- CSV helpers ----------


def _flatten_report(report: dict) -> dict:
    """
    Extract key metrics from sklearn classification_report(output_dict=True).
    For binary decomposition, we capture both per-class and averaged metrics.
    """
    out = {}
    # macro / weighted
    for k in ("macro avg", "weighted avg"):
        if k in report:
            out[f"{k.replace(' ', '_')}_precision"] = report[k].get("precision", "")
            out[f"{k.replace(' ', '_')}_recall"] = report[k].get("recall", "")
            out[f"{k.replace(' ', '_')}_f1"] = report[k].get("f1-score", "")
    # per-class (for binary: 'target', 'negative')
    for k in report:
        if k in ("accuracy", "macro avg", "weighted avg"):
            continue
        if isinstance(report[k], dict):
            tag = k.replace(" ", "_")
            out[f"{tag}_precision"] = report[k].get("precision", "")
            out[f"{tag}_recall"] = report[k].get("recall", "")
            out[f"{tag}_f1"] = report[k].get("f1-score", "")
            out[f"{tag}_support"] = report[k].get("support", "")
    # top-level accuracy
    if "accuracy" in report:
        out["accuracy"] = report["accuracy"]
    return out


def _write_csv(path: Path, rows: list[dict]):
    """Write rows to CSV with a stable header across heterogeneous dicts."""
    if not rows:
        return
    # union of keys preserving first-seen order
    fieldnames, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ---------- Core runners ----------


def _prepare_adv_mix_loaders(
    model,
    cfg,
    device,
    calib_dl,
    *,
    eps_start_nlp: float,
    eps_max_nlp: float,
    eps_step_nlp: float,
    eps_start_vision: float,
    eps_max_vision: float,
    eps_step_vision: float,
    adv_ratio: float,
    vision_stats: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
) -> DataLoader:
    """
    Create a 'pruning input' DataLoader according to adv_ratio.
    For NLP, both base and adv are in inputs_embeds form so that Wanda/LB-Mask
    see a homogeneous input structure.
    """
    if is_vision(cfg):
        items = []
        for batch in calib_dl:
            px = batch["pixel_values"]
            y = batch["labels"]
            for i in range(px.size(0)):
                items.append({"pixel_values": px[i], "labels": y[i]})
        base_ds = DictListDataset(items)

        mean, std = vision_stats or (None, None)
        adv_ds = vision_adv_dataset(
            model,
            calib_dl,
            device,
            eps_start_px=eps_start_vision,
            eps_max_px=eps_max_vision,
            eps_step_px=eps_step_vision,
            mean=mean,
            std=std,
        )

        mixed = make_mixed_loader(
            base_ds=base_ds,
            adv_ds=adv_ds,
            adv_ratio=adv_ratio,
            batch_size=8,
            shuffle=True,
        )
        return mixed

    else:
        # NLP path
        base_ds = nlp_to_embeds_dataset(model, calib_dl, device, max_batches=None)
        adv_ds = nlp_adv_dataset(
            model,
            calib_dl,
            device,
            eps_start=eps_start_nlp,
            eps_max=eps_max_nlp,
            eps_step=eps_step_nlp,
        )
        mixed = make_mixed_loader(
            base_ds=base_ds,
            adv_ds=adv_ds,
            adv_ratio=adv_ratio,
            batch_size=8,
            shuffle=True,
        )

    return mixed


def _prune_with_method(
    base_model,
    method: str,
    *,
    calib_or_train_dl: DataLoader,
    device: torch.device,
    sparsity: float,
):
    """
    Return a pruned COPY of base_model using the chosen method.
    - "wanda": one-shot, activation-aware pruning (no training).
    - "lbmask": learn binary masks with frozen base weights, then project.
    """
    if method == "wanda":
        return wanda_prune(
            base_model,
            calib_or_train_dl,
            sparsity_ratio=sparsity,
            device=device,
            method="wanda",
            nm_values=None,
        )

    if method == "lbmask":
        mp_cfg = MaskPruneConfig(
            granularity="out",
            # Mask learning
            lr=5e-3,
            lmbda_l1=1e-3,
            num_epochs=3,
            skip_final_classifier=True,
            # Progressive pruning
            freeze_base=True,
            prune_during_train=True,
            schedule="constant",
            start_sparsity=0.0,
            end_sparsity=sparsity,
            begin_step=0,
            end_step=len(calib_or_train_dl) * 3,  # 3 epochs
            update_every=100,
            hard_apply=False,
            zero_bias_when_masked=True,
            verbose=True,
        )
        masked_model, overlays = train_with_progressive_pruning(
            base_model=base_model,
            train_dl=calib_or_train_dl,
            mp_cfg=mp_cfg,
            base_lr=0.0,  # base weights are frozen
            num_epochs=mp_cfg.num_epochs,
            device=device,
        )
        pruned = export_pruned_copy(base_model, overlays).to(device)
        return pruned

    raise ValueError(f"Unknown method: {method}")


def run_decompose_prune_once(
    cfg: ExpConfig,
    *,
    methods: List[str],
    adv_ratio: float,
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
    mix_loader_calib = _prepare_adv_mix_loaders(
        model,
        cfg,
        device,
        calib_dl,
        eps_start_nlp=eps_start_nlp,
        eps_max_nlp=eps_max_nlp,
        eps_step_nlp=eps_step_nlp,
        eps_start_vision=eps_start_vision,
        eps_max_vision=eps_max_vision,
        eps_step_vision=eps_step_vision,
        adv_ratio=adv_ratio,
        vision_stats=vision_stats,
    )

    mix_loader_train = _prepare_adv_mix_loaders(
        model,
        cfg,
        device,
        train_dl,
        eps_start_nlp=eps_start_nlp,
        eps_max_nlp=eps_max_nlp,
        eps_step_nlp=eps_step_nlp,
        eps_start_vision=eps_start_vision,
        eps_max_vision=eps_max_vision,
        eps_step_vision=eps_step_vision,
        adv_ratio=adv_ratio,
        vision_stats=vision_stats,
    )

    results = []
    all_csv_rows = []

    for method in methods:
        # LB-Mask -> use full-train mix; Wanda -> use calibration mix
        use_loader = mix_loader_train if method == "lbmask" else mix_loader_calib

        pruned = _prune_with_method(
            base_model=model,
            method=method,
            calib_or_train_dl=use_loader,
            device=device,
            sparsity=sparsity,
        )
        overall_sparsity, _ = calc_sparsity(pruned)
        acc_orig, _ = evaluate(pruned, test_dl, cfg, device)

        # Adversarial test loader & accuracy
        if is_vision(cfg):
            mean, std = vision_stats or (None, None)
            adv_test_ds = vision_adv_dataset(
                pruned,
                test_dl,
                device,
                eps_start_px=eps_start_vision,
                eps_max_px=eps_max_vision,
                eps_step_px=eps_step_vision,
                mean=mean,
                std=std,
            )
        else:
            adv_test_ds = nlp_adv_dataset(
                pruned,
                test_dl,
                device,
                eps_start=eps_start_nlp,
                eps_max=eps_max_nlp,
                eps_step=eps_step_nlp,
            )

        adv_test_loader = DataLoader(adv_test_ds, batch_size=cfg.batch_size)
        robust_acc, _ = evaluate(pruned, adv_test_loader, cfg, device)

        row = {
            "method": method,
            "adv_ratio": adv_ratio,
            "sparsity_target": sparsity,
            "sparsity_achieved_pct": overall_sparsity,
            "acc_original": acc_orig,
            "acc_adversarial": robust_acc,
        }

        if target_class is not None:
            # Original (binary)
            decomp = decompose_to_target_vs_rest(pruned, target_class).to(device)
            bin_acc_orig, rep_orig = evaluate(
                decomp, test_dl, cfg, device, binary_target_idx=target_class
            )
            r0 = {
                "eval_set": "original",
                "method": method,
                "adv_ratio": adv_ratio,
                "sparsity_target": sparsity,
                "sparsity_achieved_pct": overall_sparsity,
                "binary_target": int(target_class),
            }
            r0.update(_flatten_report(rep_orig))
            all_csv_rows.append(r0)

            # Adversarial (binary)
            if is_vision(cfg):
                mean, std = vision_stats or (None, None)
                adv_test_ds_bin = vision_adv_dataset(
                    decomp,
                    test_dl,
                    device,
                    eps_start_px=eps_start_vision,
                    eps_max_px=eps_max_vision,
                    eps_step_px=eps_step_vision,
                    mean=mean,
                    std=std,
                    binary_target_idx=target_class,
                )
            else:
                adv_test_ds_bin = nlp_adv_dataset(
                    decomp,
                    test_dl,
                    device,
                    eps_start=eps_start_nlp,
                    eps_max=eps_max_nlp,
                    eps_step=eps_step_nlp,
                    binary_target_idx=target_class,
                )
            adv_test_loader_bin = DataLoader(adv_test_ds_bin, batch_size=cfg.batch_size)
            bin_acc_adv, rep_adv = evaluate(
                decomp, adv_test_loader_bin, cfg, device, binary_target_idx=target_class
            )

            r1 = {
                "eval_set": "adversarial",
                "method": method,
                "adv_ratio": adv_ratio,
                "sparsity_target": sparsity,
                "sparsity_achieved_pct": overall_sparsity,
                "binary_target": int(target_class),
            }
            r1.update(_flatten_report(rep_adv))
            all_csv_rows.append(r1)

            row.update(
                {
                    "binary_target": int(target_class),
                    "binary_acc_original": bin_acc_orig,
                    "binary_acc_adversarial": bin_acc_adv,
                }
            )

        results.append(row)

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
    parser.add_argument("--methods", nargs="+", default=["wanda", "lbmask"])
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

    # Build base experiment config using main.py presets
    from main import build_default_bert, build_default_vit

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

    # Save a single CSV per run setup (across methods x ratios) for the decomposed model
    if all_csv_rows:
        from pathlib import Path

        # Save under: <outdir>/<exp>_<size>_<dataset>/target_<id>.csv
        outdir = Path(args.outdir) / f"{args.exp}_{args.model_size}_{args.dataset}"
        outdir.mkdir(parents=True, exist_ok=True)

        # If no target class (tgt is None), save as 'all.csv'
        tname = "all" if tgt is None else f"target_{int(tgt):02d}"
        out_csv = outdir / f"{tname}.csv"

        _write_csv(out_csv, all_csv_rows)
        print(f"[CSV] Saved: {out_csv.resolve()}")

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
