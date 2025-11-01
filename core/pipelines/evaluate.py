# pipelines/evaluate.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch
from torch.utils.data import DataLoader
from core.config import ExpConfig
from core.evaluation import evaluate
from core.utils.metrics_csv import flatten_report, write_csv


def evaluate_to_csv(
    model,
    loader: DataLoader,
    cfg: ExpConfig,
    device: torch.device,
    *,
    out_csv: Path,
    binary_target_idx: Optional[int] = None,
) -> Tuple[float, Dict]:
    """
    Run evaluation and save F1/Recall/Precision(+Accuracy) to CSV.
    The CSV will contain per-class, macro, weighted metrics (flattened).
    """
    acc, report = evaluate(
        model, loader, cfg, device, binary_target_idx=binary_target_idx
    )
    rows = []
    flat = flatten_report(report)
    flat.update(
        {
            "binary_target": (
                int(binary_target_idx) if binary_target_idx is not None else ""
            ),
        }
    )
    rows.append(flat)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows)
    return acc, report
