# core/utils/metrics_csv.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import csv


def flatten_report(rep: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten sklearn classification_report dict to a single row.
    Keys will look like:
      - class_<name>_precision / recall / f1-score / support
      - macro_avg_* , weighted_avg_* , accuracy
    """
    out: Dict[str, Any] = {}
    for k, v in rep.items():
        if isinstance(v, dict):
            for m, val in v.items():
                out[f"{k}_{m}".replace(" ", "_")] = val
        else:
            out[k] = v
    # Provide stable aliases for known sections
    # e.g., target, negative, macro avg, weighted avg
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Union of all keys
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    header = sorted(keys)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
