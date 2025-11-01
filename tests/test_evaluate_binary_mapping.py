import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import types
from core.evaluation import evaluate
from core.config import ExpConfig


class TinyBinHead(nn.Module):
    def __init__(self, in_f=4):
        super().__init__()
        self.backbone = nn.Linear(in_f, 8)
        self.classifier = nn.Linear(8, 2)
        self.config = types.SimpleNamespace(num_labels=2)
        self.num_labels = 2

    def forward(self, input_ids=None, pixel_values=None, labels=None, **kw):
        x = input_ids if input_ids is not None else pixel_values
        h = self.backbone(x.float())
        return types.SimpleNamespace(logits=self.classifier(h), loss=None)


def _dl(n=64, d=4, target_idx=1):
    x = torch.randn(n, d)
    # ground truth multiclass labels in {0,1,2}; map later to {0:target,1:neg}
    y_mc = torch.randint(0, 3, (n,))
    ds = [{"input_ids": x[i], "labels": y_mc[i]} for i in range(n)]

    class ToyDS(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return ds[i]

    return DataLoader(ToyDS(), batch_size=16), y_mc, target_idx


def test_binary_eval_runs():
    cfg = ExpConfig(
        model_name="", dataset_name="ag_news", task_type="nlp", num_labels=3
    )
    device = torch.device("cpu")
    dl, _, t = _dl()
    model = TinyBinHead()
    # Eval under binary mapping (0=target, 1=negative)
    acc, rep = evaluate(model, dl, cfg, device, binary_target_idx=t)
    assert "accuracy" in rep and "target" in rep and "negative" in rep
