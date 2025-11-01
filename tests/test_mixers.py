# tests/test_mixers.py
import torch
from torch.utils.data import Dataset
from core.adversarial.mixers import make_mixed_loader


class ToyDS(Dataset):
    def __init__(self, n, label):
        self.x = torch.arange(n).float().unsqueeze(-1)
        self.y = torch.full((n,), label, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"feat": self.x[i], "labels": self.y[i]}


def test_adv_ratio_approx():
    base, adv = ToyDS(100, 0), ToyDS(100, 1)
    loader = make_mixed_loader(
        base, adv, batch_size=20, adv_ratio=0.25, num_workers=0, pin_memory=False
    )
    for batch in loader:
        y = batch["labels"]
        adv_cnt = (y == 1).sum().item()
        assert 3 <= adv_cnt <= 7  # around 5 (25% of 20), allow slack


# tests/test_generators_api.py
def test_dictsampledataset_shape():
    from core.adversarial.generators import DictSampleDataset

    items = [{"a": torch.randn(3), "labels": torch.tensor(1)} for _ in range(5)]
    ds = DictSampleDataset(items)
    assert len(ds) == 5
    assert set(ds[0].keys()) == {"a", "labels"}
