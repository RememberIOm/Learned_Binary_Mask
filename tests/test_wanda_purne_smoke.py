import torch
from torch.utils.data import DataLoader
from core.pruning.wanda_pruning import wanda_prune


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(6, 6)
        self.classifier = torch.nn.Linear(6, 3)  # will be skipped by Wanda

    def forward(self, input_ids=None, labels=None, **kw):
        h = self.encoder(input_ids.float())
        return type("O", (), {"logits": self.classifier(h)})


def _calib(n=32, d=6):
    x = torch.randn(n, d)
    y = torch.randint(0, 3, (n,))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"input_ids": x[i], "labels": y[i]}

    return DataLoader(DS(), batch_size=8)


def test_wanda_prune_masks_some_weights():
    model = Toy()
    pruned = wanda_prune(
        model, _calib(), sparsity_ratio=0.5, device=torch.device("cpu")
    )
    w = pruned.encoder.weight
    # Expect roughly half entries zero per-row (allow slack)
    zero_frac = (w == 0).float().mean().item()
    assert 0.2 <= zero_frac <= 0.8
