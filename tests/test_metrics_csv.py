from pathlib import Path
from core.utils.metrics_csv import flatten_report, write_csv


def test_flatten_and_write(tmp_path: Path):
    rep = {
        "accuracy": 0.9,
        "target": {"precision": 1.0, "recall": 0.8, "f1-score": 0.89, "support": 10},
        "negative": {
            "precision": 0.85,
            "recall": 0.95,
            "f1-score": 0.90,
            "support": 12,
        },
        "macro avg": {
            "precision": 0.925,
            "recall": 0.875,
            "f1-score": 0.895,
            "support": 22,
        },
        "weighted avg": {
            "precision": 0.92,
            "recall": 0.9,
            "f1-score": 0.90,
            "support": 22,
        },
    }
    row = flatten_report(rep)
    assert "accuracy" in row and "target_precision" in row
    out = tmp_path / "m.csv"
    write_csv(out, [row])
    assert out.exists() and out.read_text().count("\n") >= 2
