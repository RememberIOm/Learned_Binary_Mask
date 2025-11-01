# core/model_utils.py

from __future__ import annotations

from pathlib import Path
import os

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
)

from core.config import ExpConfig


def get_device(cfg: ExpConfig) -> torch.device:
    """Pick a device based on availability and cfg.gpu."""
    if torch.cuda.is_available():
        if cfg.gpu is None or int(cfg.gpu) < 0:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(cfg.gpu)}")
    return torch.device("cpu")


def load_model_and_processor(cfg: ExpConfig):
    """
    Load HF model + pre/post processor with finetuned model discovery under ./models.
    Priority:
      1) Load from cfg.save_dir if it looks like a HF directory (config.json etc.)
      2) Else, if a single .pt state_dict is found in cfg.save_dir, load it onto a base model
      3) Else, fall back to hub model_name
    For ViT, we replace the classifier head to match num_labels.
    """

    def _try_load_from_saved_dir_nlp():
        sd = Path(cfg.save_dir)
        if not sd.exists():
            return None
        # HF-style directory?
        if (sd / "config.json").exists():
            tok = AutoTokenizer.from_pretrained(sd)
            mdl = AutoModelForSequenceClassification.from_pretrained(sd)
            return mdl, tok
        # Single .pt state_dict?
        pts = list(sd.glob("*.pt"))
        if pts:
            tok = AutoTokenizer.from_pretrained(cfg.model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name, num_labels=cfg.num_labels
            )
            state = torch.load(pts[0], map_location="cpu")
            mdl.load_state_dict(state, strict=False)
            return mdl, tok
        return None

    def _try_load_from_saved_dir_vision():
        sd = Path(cfg.save_dir)
        if not sd.exists():
            return None
        if (sd / "config.json").exists():
            ip = AutoImageProcessor.from_pretrained(sd)
            mdl = AutoModelForImageClassification.from_pretrained(sd)
            return mdl, ip
        pts = list(sd.glob("*.pt"))
        if pts:
            ip = AutoImageProcessor.from_pretrained(cfg.model_name)
            mdl = AutoModelForImageClassification.from_pretrained(
                cfg.model_name,
                num_labels=cfg.num_labels,
                ignore_mismatched_sizes=True,
            )
            # Align classifier head, then load weights
            if hasattr(mdl, "classifier") and isinstance(mdl.classifier, nn.Linear):
                mdl.classifier = nn.Linear(mdl.classifier.in_features, cfg.num_labels)
            state = torch.load(pts[0], map_location="cpu")
            mdl.load_state_dict(state, strict=False)
            return mdl, ip
        return None

    if cfg.task_type == "nlp":
        loaded = _try_load_from_saved_dir_nlp()
        if loaded is not None:
            model, tokenizer = loaded
            # ensure label dim
            if hasattr(model, "num_labels"):
                model.num_labels = cfg.num_labels
            if hasattr(model, "config") and hasattr(model.config, "num_labels"):
                model.config.num_labels = cfg.num_labels
            return model, tokenizer
        # Fallback to hub
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=cfg.num_labels
        )
        return model, tokenizer

    # Vision
    loaded = _try_load_from_saved_dir_vision()
    if loaded is not None:
        model, image_processor = loaded
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
            # keep classifier size consistent with cfg
            if model.classifier.out_features != cfg.num_labels:
                model.classifier = nn.Linear(
                    model.classifier.in_features, cfg.num_labels
                )
        return model, image_processor

    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name)
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, cfg.num_labels)
    return model, image_processor
