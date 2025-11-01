import torch
from torch.utils.data import DataLoader
from core.pipelines.adv_data import (
    build_mixed_pruning_loader,
    build_adversarial_test_loader,
)
from core.config import build_default_bert


class ToyModel(torch.nn.Module):
    def __init__(self, d=8, k=3):
        super().__init__()
        self.emb = torch.nn.Embedding(100, d)
        self.classifier = torch.nn.Linear(d, k)
        self.config = type("C", (), {"num_labels": k})()
        self.num_labels = k

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kw):
        x = self.emb(input_ids) if input_ids is not None else inputs_embeds
        h = x.mean(dim=1)
        return type(
            "O",
            (),
            {
                "logits": self.classifier(h),
                "loss": (self.classifier(h).mean() if labels is not None else None),
            },
        )


def _dummy_loader(n=32, t=10, vocab=100):
    ids = torch.randint(0, vocab, (n, t))
    y = torch.randint(0, 3, (n,))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"input_ids": ids[i], "labels": y[i]}

    return DataLoader(DS(), batch_size=8)


def test_builders_run():
    cfg = build_default_bert(
        gpu=None,
        prune_method="wanda",
        model_size="tiny",
        sparsity=0.5,
        dataset="ag_news",
    )
    device = torch.device("cpu")
    model = ToyModel()
    base = _dummy_loader()
    mix = build_mixed_pruning_loader(
        model,
        cfg,
        base,
        device,
        adv_ratio=0.5,
        eps_nlp={"start": 0.0, "max": 0.1, "step": 0.1},
        eps_vision={"start_px": 0.0, "max_px": 8 / 255, "step_px": 2 / 255},
    )
    assert next(iter(mix))  # has at least one batch
    adv = build_adversarial_test_loader(
        model,
        base,
        cfg,
        device,
        eps_nlp={"start": 0.0, "max": 0.1, "step": 0.1},
        eps_vision={"start_px": 0.0, "max_px": 8 / 255, "step_px": 2 / 255},
    )
    assert next(iter(adv))  # has at least one batch
