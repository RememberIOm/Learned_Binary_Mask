# utils/metrics_csv.py
from __future__ import annotations
import csv
from pathlib import Path


def flatten_report(report: dict) -> dict:
    """Flatten sklearn classification_report(output_dict=True) into one-row dict."""
    out = {}
    for k in ("macro avg", "weighted avg"):
        if k in report:
            out[f"{k.replace(' ', '_')}_precision"] = report[k].get("precision", "")
            out[f"{k.replace(' ', '_')}_recall"] = report[k].get("recall", "")
            out[f"{k.replace(' ', '_')}_f1"] = report[k].get("f1-score", "")
    for k, v in report.items():
        if k in ("accuracy", "macro avg", "weighted avg") or not isinstance(v, dict):
            continue
        tag = k.replace(" ", "_")
        out[f"{tag}_precision"] = v.get("precision", "")
        out[f"{tag}_recall"] = v.get("recall", "")
        out[f"{tag}_f1"] = v.get("f1-score", "")
        out[f"{tag}_support"] = v.get("support", "")
    if "accuracy" in report:
        out["accuracy"] = report["accuracy"]
    return out


def write_csv(path: Path, rows: list[dict]):
    """Write heterogeneous dict rows with stable union header."""
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
