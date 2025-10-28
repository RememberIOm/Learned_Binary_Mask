# model_utils.py

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
)

from config import ExpConfig


def get_device(cfg: ExpConfig) -> torch.device:
    """Pick a device based on availability and cfg.gpu."""
    if torch.cuda.is_available():
        if cfg.gpu is None or int(cfg.gpu) < 0:
            return torch.device("cuda")
        return torch.device(f"cuda:{int(cfg.gpu)}")
    return torch.device("cpu")


def load_model_and_processor(cfg: ExpConfig):
    """
    Load HF model + pre/post processor based on task type.
    For ViT, we replace the classifier head to match num_labels.
    """
    if cfg.task_type == "nlp":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=cfg.num_labels
        )
        return model, tokenizer

    # Vision
    image_processor = AutoImageProcessor.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, cfg.num_labels)
    return model, image_processor
