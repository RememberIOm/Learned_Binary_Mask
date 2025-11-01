# cli/run.py

from __future__ import annotations
import argparse
from pathlib import Path

from core.config import build_default_bert, build_default_vit
from core.model_utils import get_device, load_model_and_processor
from core.data_utils import is_vision, prepare_dataloaders
from core.pipelines.pruning import prune_model
from core.decomposition import decompose_binary
from core.pipelines.evaluate import evaluate_to_csv
from core.pipelines.adv_data import (
    build_mixed_pruning_loader,
    build_adversarial_test_loader,
)
from core.bootstrap_training import train_initial_model


def _build_cfg(exp, model_size, dataset, method, sparsity):
    build = build_default_bert if exp == "bert" else build_default_vit
    return build(0, method, model_size, sparsity, dataset)


def _ensure_model(cfg):
    device = get_device(cfg)
    model, processor = load_model_and_processor(cfg)
    train_dl, test_dl, calib_dl = prepare_dataloaders(cfg, processor)
    return device, model, (train_dl, test_dl, calib_dl)


def cmd_decompose_original(args):
    cfg = _build_cfg(args.exp, args.model_size, args.dataset, "wanda", 0.0)
    device, model, (train_dl, test_dl, _) = _ensure_model(cfg)
    model_bin = decompose_binary(model, args.target).to(device)
    out = (
        Path(args.outdir)
        / f"{args.exp}_{args.model_size}_{args.dataset}"
        / f"target_{args.target:02d}"
    )
    if args.eval in ("original", "both"):
        evaluate_to_csv(
            model_bin,
            test_dl,
            cfg,
            device,
            out_csv=out / "binary_original.csv",
            binary_target_idx=args.target,
        )
    if args.eval in ("adversarial", "both"):
        adv_dl = build_adversarial_test_loader(
            model_bin,
            test_dl,
            cfg,
            device,
            eps_nlp=dict(
                start=args.eps_start_nlp,
                max=args.eps_max_nlp,
                step=args.eps_step_nlp,
            ),
            eps_vision=dict(
                start_px=args.eps_start_vision,
                max_px=args.eps_max_vision,
                step_px=args.eps_step_vision,
            ),
            vision_stats=None,
            binary_target_idx=args.target,
        )
        evaluate_to_csv(
            model_bin,
            adv_dl,
            cfg,
            device,
            out_csv=out / "binary_adversarial.csv",
            binary_target_idx=args.target,
        )


def cmd_decompose_mixed(args):
    cfg = _build_cfg(
        args.exp, args.model_size, args.dataset, args.methods[0], args.sparsity
    )
    device, model, (train_dl, test_dl, calib_dl) = _ensure_model(cfg)
    for ratio in args.ratios:
        for method in args.methods:
            use_loader = build_mixed_pruning_loader(
                model,
                cfg,
                (train_dl if method == "lbmask" else calib_dl),
                device,
                ratio,
                eps_nlp=dict(
                    start=args.eps_start_nlp,
                    max=args.eps_max_nlp,
                    step=args.eps_step_nlp,
                ),
                eps_vision=dict(
                    start=args.eps_start_vision,
                    max=args.eps_max_vision,
                    step=args.eps_step_vision,
                ),
                vision_stats=None,
            )
            pruned, _ = prune_model(
                model,
                method=method,
                loader=use_loader,
                device=device,
                sparsity=args.sparsity,
            )
            model_bin = decompose_binary(pruned, args.target).to(device)
            base = (
                Path(args.outdir)
                / f"{args.exp}_{args.model_size}_{args.dataset}"
                / method
                / f"ratio_{int(ratio*100):03d}"
                / f"target_{args.target:02d}"
            )
            if args.eval in ("original", "both"):
                evaluate_to_csv(
                    model_bin,
                    test_dl,
                    cfg,
                    device,
                    out_csv=base / "binary_original.csv",
                    binary_target_idx=args.target,
                )
            if args.eval in ("adversarial", "both"):
                adv_dl = build_adversarial_test_loader(
                    model_bin,
                    test_dl,
                    cfg,
                    device,
                    eps_nlp=dict(
                        start=args.eps_start_nlp,
                        max=args.eps_max_nlp,
                        step=args.eps_step_nlp,
                    ),
                    eps_vision=dict(
                        start=args.eps_start_vision,
                        max=args.eps_max_vision,
                        step=args.eps_step_vision,
                    ),
                    vision_stats=None,
                    binary_target_idx=args.target,
                )
                evaluate_to_csv(
                    model_bin,
                    adv_dl,
                    cfg,
                    device,
                    out_csv=base / "binary_adversarial.csv",
                    binary_target_idx=args.target,
                )


def cmd_prune_eval(args):
    cfg = _build_cfg(
        args.exp, args.model_size, args.dataset, args.methods[0], args.sparsity
    )
    device, model, (train_dl, test_dl, calib_dl) = _ensure_model(cfg)
    for method in args.methods:
        for ratio in args.ratios:
            mix_calib = build_mixed_pruning_loader(
                model,
                cfg,
                calib_dl,
                device,
                ratio,
                eps_nlp=dict(
                    start=args.eps_start_nlp,
                    max=args.eps_max_nlp,
                    step=args.eps_step_nlp,
                ),
                eps_vision=dict(
                    start=args.eps_start_vision,
                    max=args.eps_max_vision,
                    step=args.eps_step_vision,
                ),
            )
            mix_train = build_mixed_pruning_loader(
                model,
                cfg,
                train_dl,
                device,
                ratio,
                eps_nlp=dict(
                    start=args.eps_start_nlp,
                    max=args.eps_max_nlp,
                    step=args.eps_step_nlp,
                ),
                eps_vision=dict(
                    start=args.eps_start_vision,
                    max=args.eps_max_vision,
                    step=args.eps_step_vision,
                ),
            )
            use_loader = mix_train if method == "lbmask" else mix_calib
            pruned, _ = prune_model(
                model,
                method=method,
                loader=use_loader,
                device=device,
                sparsity=args.sparsity,
            )
            out = (
                Path(args.outdir)
                / f"{args.exp}_{args.model_size}_{args.dataset}"
                / method
                / f"ratio_{int(ratio*100):03d}"
            )
            if args.eval in ("original", "both"):
                evaluate_to_csv(
                    pruned,
                    test_dl,
                    cfg,
                    device,
                    out_csv=out / "multiclass_original.csv",
                    binary_target_idx=None,
                )
            if args.eval in ("adversarial", "both"):
                adv_dl = build_adversarial_test_loader(
                    pruned,
                    test_dl,
                    cfg,
                    device,
                    eps_nlp=dict(
                        start=args.eps_start_nlp,
                        max=args.eps_max_nlp,
                        step=args.eps_step_nlp,
                    ),
                    eps_vision=dict(
                        start=args.eps_start_vision,
                        max=args.eps_max_vision,
                        step=args.eps_step_vision,
                    ),
                )
                evaluate_to_csv(
                    pruned,
                    adv_dl,
                    cfg,
                    device,
                    out_csv=out / "multiclass_adversarial.csv",
                    binary_target_idx=None,
                )


def cmd_prune_then_decompose(args):
    # Prune with chosen method/ratio, then decompose per target and eval
    cfg = _build_cfg(
        args.exp, args.model_size, args.dataset, args.methods[0], args.sparsity
    )
    device, model, (train_dl, test_dl, calib_dl) = _ensure_model(cfg)
    for t in args.targets:
        for method in args.methods:
            for ratio in args.ratios:
                mix_calib = build_mixed_pruning_loader(
                    model,
                    cfg,
                    calib_dl,
                    device,
                    ratio,
                    eps_nlp=dict(
                        start=args.eps_start_nlp,
                        max=args.eps_max_nlp,
                        step=args.eps_step_nlp,
                    ),
                    eps_vision=dict(
                        start=args.eps_start_vision,
                        max=args.eps_max_vision,
                        step=args.eps_step_vision,
                    ),
                )
                mix_train = build_mixed_pruning_loader(
                    model,
                    cfg,
                    train_dl,
                    device,
                    ratio,
                    eps_nlp=dict(
                        start=args.eps_start_nlp,
                        max=args.eps_max_nlp,
                        step=args.eps_step_nlp,
                    ),
                    eps_vision=dict(
                        start=args.eps_start_vision,
                        max=args.eps_max_vision,
                        step=args.eps_step_vision,
                    ),
                )
                use_loader = mix_train if method == "lbmask" else mix_calib
                pruned, _ = prune_model(
                    model,
                    method=method,
                    loader=use_loader,
                    device=device,
                    sparsity=args.sparsity,
                )
                model_bin = decompose_binary(pruned, t).to(device)
                base = (
                    Path(args.outdir)
                    / f"{args.exp}_{args.model_size}_{args.dataset}"
                    / method
                    / f"ratio_{int(ratio*100):03d}"
                    / f"target_{t:02d}"
                )
                if args.eval in ("original", "both"):
                    evaluate_to_csv(
                        model_bin,
                        test_dl,
                        cfg,
                        device,
                        out_csv=base / "binary_original.csv",
                        binary_target_idx=t,
                    )
                if args.eval in ("adversarial", "both"):
                    adv_dl = build_adversarial_test_loader(
                        model_bin,
                        test_dl,
                        cfg,
                        device,
                        eps_nlp=dict(
                            start=args.eps_start_nlp,
                            max=args.eps_max_nlp,
                            step=args.eps_step_nlp,
                        ),
                        eps_vision=dict(
                            start=args.eps_start_vision,
                            max=args.eps_max_vision,
                            step=args.eps_step_vision,
                        ),
                        binary_target_idx=t,
                    )
                    evaluate_to_csv(
                        model_bin,
                        adv_dl,
                        cfg,
                        device,
                        out_csv=base / "binary_adversarial.csv",
                        binary_target_idx=t,
                    )


def main():
    ap = argparse.ArgumentParser(description="Unified runner for 4 experiment types")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_eps(a):
        a.add_argument("--eps-start-nlp", type=float, default=0.0)
        a.add_argument("--eps-max-nlp", type=float, default=0.25)
        a.add_argument("--eps-step-nlp", type=float, default=0.01)
        a.add_argument("--eps-start-vision", type=float, default=0.0)
        a.add_argument("--eps-max-vision", type=float, default=8 / 255)
        a.add_argument("--eps-step-vision", type=float, default=2 / 255)

    # 0) bootstrap-train: create initial task model checkpoint (no use after pruning)
    b0 = sub.add_parser("bootstrap-train")
    b0.add_argument("--exp", choices=["bert", "vit"], default="bert")
    b0.add_argument("--model_size", choices=["tiny", "small"], default="small")
    b0.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    b0.add_argument("--epochs", type=int, default=3)
    b0.add_argument("--lr", type=float, default=5e-5)
    b0.add_argument("--freeze-backbone", action="store_true", default=True)
    b0.add_argument("--outdir", type=str, default="./models/checkpoints_bootstrap")

    def _cmd_bootstrap(args):
        cfg = _build_cfg(args.exp, args.model_size, args.dataset, "wanda", 0.0)
        device, _, _ = _ensure_model(cfg)
        ckpt = train_initial_model(
            cfg,
            device,
            epochs=args.epochs,
            lr=args.lr,
            freeze_backbone=args.freeze_backbone,
            outdir=args.outdir,
        )
        print(f"[bootstrap] saved: {ckpt}")

    b0.set_defaults(func=_cmd_bootstrap)

    # 1) decompose-original
    d1 = sub.add_parser("decompose-original")
    d1.add_argument("--exp", choices=["bert", "vit"], default="bert")
    d1.add_argument("--model_size", choices=["tiny", "small"], default="small")
    d1.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    d1.add_argument("--target", type=int, default=0)
    d1.add_argument(
        "--eval", choices=["original", "adversarial", "both"], default="both"
    )
    d1.add_argument("--outdir", type=str, default="./results_decompose_original")
    add_eps(d1)
    d1.set_defaults(func=cmd_decompose_original)

    # 2) decompose-mixed
    d2 = sub.add_parser("decompose-mixed")
    d2.add_argument("--exp", choices=["bert", "vit"], default="bert")
    d2.add_argument("--model_size", choices=["tiny", "small"], default="small")
    d2.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    d2.add_argument("--methods", nargs="+", default=["wanda", "lbmask", "mi"])
    d2.add_argument(
        "--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    d2.add_argument("--sparsity", type=float, default=0.5)
    d2.add_argument("--target", type=int, default=0)
    d2.add_argument(
        "--eval", choices=["original", "adversarial", "both"], default="both"
    )
    d2.add_argument("--outdir", type=str, default="./results_decompose_mixed")
    add_eps(d2)
    d2.set_defaults(func=cmd_decompose_mixed)

    # 3) prune-eval (multiclass)
    d3 = sub.add_parser("prune-eval")
    d3.add_argument("--exp", choices=["bert", "vit"], default="bert")
    d3.add_argument("--model_size", choices=["tiny", "small"], default="small")
    d3.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    d3.add_argument("--methods", nargs="+", default=["wanda", "lbmask", "mi"])
    d3.add_argument(
        "--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    d3.add_argument("--sparsity", type=float, default=0.5)
    d3.add_argument(
        "--eval", choices=["original", "adversarial", "both"], default="both"
    )
    d3.add_argument("--outdir", type=str, default="./results_prune_eval")
    add_eps(d3)
    d3.set_defaults(func=cmd_prune_eval)

    # 4) prune-then-decompose (binary)
    d4 = sub.add_parser("prune-then-decompose")
    d4.add_argument("--exp", choices=["bert", "vit"], default="bert")
    d4.add_argument("--model_size", choices=["tiny", "small"], default="small")
    d4.add_argument(
        "--dataset",
        choices=["ag_news", "dbpedia_14", "cifar10", "fashion_mnist"],
        default="ag_news",
    )
    d4.add_argument("--methods", nargs="+", default=["wanda", "lbmask", "mi"])
    d4.add_argument(
        "--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0]
    )
    d4.add_argument("--sparsity", type=float, default=0.5)
    d4.add_argument("--targets", nargs="+", type=int, default=[0])
    d4.add_argument(
        "--eval", choices=["original", "adversarial", "both"], default="both"
    )
    d4.add_argument("--outdir", type=str, default="./results_prune_then_decompose")
    add_eps(d4)
    d4.set_defaults(func=cmd_prune_then_decompose)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
