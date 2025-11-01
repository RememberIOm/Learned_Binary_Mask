# data_utils.py

from __future__ import annotations
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import load_dataset

from core.config import ExpConfig


def class_names_for(dataset_name: str) -> Optional[List[str]]:
    """Return human-friendly class names when available."""
    if dataset_name == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    if dataset_name == "fashion_mnist":
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    if dataset_name == "ag_news":
        return ["World", "Sports", "Business", "Sci/Tech"]
    if dataset_name == "dbpedia_14":
        return [
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ]
    return None


def is_vision(cfg: ExpConfig) -> bool:
    return cfg.task_type == "vision"


def prepare_dataloaders(cfg: ExpConfig, processor):
    """
    Build train/test/calibration DataLoaders.
    Output batches are dicts with unified keys:
      - NLP: {input_ids, attention_mask, labels}
      - Vision: {pixel_values, labels}
    """
    if cfg.task_type == "nlp":
        if cfg.dataset_name == "ag_news":
            ds = load_dataset(cfg.dataset_name)
        elif cfg.dataset_name == "dbpedia_14":
            ds = load_dataset("dbpedia_14")

            # DBpedia: combine title + content → text
            def _build_text(ex):
                title = ex.get("title", "") or ""
                content = ex.get("content", "") or ""
                ex["text"] = " ".join([s for s in (title, content) if s]).strip()
                return ex

            ds = ds.map(_build_text)
        else:
            raise ValueError(f"Unknown NLP dataset: {cfg.dataset_name}")

        def tok_fn(ex):
            # Tokenize prepared 'text'
            return processor(
                ex["text"],
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length,
            )

        tokenized = ds.map(tok_fn, batched=True)

        # Drop raw text-ish columns (keep only model inputs + labels)
        drop_cols = {
            "text",
            "question_title",
            "question_content",
            "best_answer",
            "title",
            "content",
            "id",
            "idx",
            "guid",
        }
        keep = [c for c in tokenized["train"].column_names if c not in drop_cols]
        tokenized = tokenized.select_columns(keep)

        # Normalize label column name
        train_cols = set(tokenized["train"].column_names)
        if "labels" in train_cols:
            pass
        elif "label" in train_cols:
            tokenized = tokenized.rename_column("label", "labels")
        elif "topic" in train_cols:
            tokenized = tokenized.rename_column("topic", "labels")
        else:
            raise ValueError(
                f"Could not find label column. Available: {sorted(train_cols)}"
            )

        tokenized.set_format("torch")

        train_ds = tokenized["train"]

        calib_idx = np.random.choice(
            len(train_ds), min(cfg.calib_nlp, len(train_ds)), replace=False
        )
        calib_ds = Subset(train_ds, calib_idx)

        test_ds = tokenized["test"]

        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        calib_dl = DataLoader(calib_ds, batch_size=8, num_workers=cfg.num_workers)
        test_dl = DataLoader(
            test_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
        return train_dl, test_dl, calib_dl

    # Vision
    size_cfg = processor.size
    tgt = (
        size_cfg
        if isinstance(size_cfg, int)
        else size_cfg.get("height") or size_cfg.get("shortest_edge") or 224
    )

    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    def to_3ch(x):
        # If grayscale, repeat to 3 channels
        return x.repeat(3, 1, 1) if x.size(0) == 1 else x

    transform = Compose(
        [
            Resize(tgt),
            ToTensor(),
            to_3ch,
            Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )

    if cfg.dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10

        train_raw = CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_raw = CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif cfg.dataset_name == "fashion_mnist":
        from torchvision.datasets import FashionMNIST

        train_raw = FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_raw = FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown vision dataset: {cfg.dataset_name}")

    # Wrap to dict-style batches for a unified forward
    class VisionDict(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __getitem__(self, idx):
            px, y = self.base[idx]
            return {"pixel_values": px, "labels": y}

        def __len__(self):
            return len(self.base)

    train_ds = VisionDict(train_raw)
    test_ds = VisionDict(test_raw)

    calib_idx = np.random.choice(
        len(train_ds), min(cfg.calib_vision, len(train_ds)), replace=False
    )
    calib_ds = Subset(train_ds, calib_idx)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)
    calib_dl = DataLoader(calib_ds, batch_size=8)
    return train_dl, test_dl, calib_dl
