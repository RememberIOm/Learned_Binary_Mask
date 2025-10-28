# evaluation.py

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from tqdm.auto import tqdm
import warnings

from config import ExpConfig
from data_utils import class_names_for
from adv_utils import map_target_vs_rest

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_dl: DataLoader,
    cfg: ExpConfig,
    device: torch.device,
    binary_target_idx: Optional[int] = None,
) -> Tuple[float, Dict]:
    """Return accuracy and full sklearn classification report dict."""
    model.eval()
    model.to(device)

    preds, labels = [], []
    for batch in tqdm(test_dl, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["labels"]
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        pred = torch.argmax(logits, dim=-1)

        if binary_target_idx is None:
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
        else:
            # Map GT labels to {0:target, 1:negative} to match TwoHeadLinear order.
            y_bin = map_target_vs_rest(y, int(binary_target_idx))
            preds.extend(pred.cpu().numpy())  # 0=target, 1=negative
            labels.extend(y_bin.cpu().numpy())

    if binary_target_idx is None:
        labels_param = list(range(cfg.num_labels))  # e.g., [0,1,2,3]
        names = class_names_for(cfg.dataset_name) or [str(i) for i in labels_param]
    else:
        labels_param = [0, 1]  # 0=target, 1=negative
        names = ["target", "negative"]

    # classification_report with fixed labels and safe zero_division
    rep_dict = classification_report(
        labels,
        preds,
        labels=labels_param,
        target_names=names,
        output_dict=True,
        digits=4,
        zero_division=0,  # avoid warnings/errors for support=0
    )

    print("\n--- Classification Report ---")
    print(
        classification_report(
            labels,
            preds,
            labels=labels_param,
            target_names=names,
            digits=4,
            zero_division=0,
        )
    )
    print("-----------------------------\n")
    return rep_dict["accuracy"], rep_dict
