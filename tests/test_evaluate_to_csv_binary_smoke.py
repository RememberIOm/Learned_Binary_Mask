from pathlib import Path
import torch, types
import torch.nn as nn
from torch.utils.data import DataLoader
from core.pipelines.evaluate import evaluate_to_csv
from core.config import ExpConfig


class TinyBin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 8)
        self.classifier = torch.nn.Linear(8, 2)
        self.config = types.SimpleNamespace(num_labels=2)
        self.num_labels = 2

    def forward(self, input_ids=None, pixel_values=None, labels=None, **kw):
        x = input_ids if input_ids is not None else pixel_values
        h = self.backbone(x.float())
        return types.SimpleNamespace(logits=self.classifier(h), loss=None)


def _dl(n=16):
    x = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"input_ids": x[i], "labels": y[i]}

    return DataLoader(DS(), batch_size=8)


def test_evaluate_to_csv_writes(tmp_path: Path):
    cfg = ExpConfig(
        model_name="", dataset_name="ag_news", task_type="nlp", num_labels=2
    )
    device = torch.device("cpu")
    model = TinyBin()
    dl = _dl()
    out = tmp_path / "report.csv"
    acc, rep = evaluate_to_csv(
        model, dl, cfg, device, out_csv=out, binary_target_idx=None
    )
    assert out.exists() and "accuracy" in rep
