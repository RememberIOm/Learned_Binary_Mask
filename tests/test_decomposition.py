import torch
import torch.nn as nn
from core.decomposition import decompose_binary


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 3)
        self.classifier = nn.Linear(3, 5)  # final head

    def forward(self, x=None, inputs_embeds=None, labels=None, pixel_values=None):
        # simple path: x -> backbone -> head
        h = self.backbone((x if x is not None else inputs_embeds))
        logits = self.classifier(h)
        return type("Out", (), {"logits": logits, "loss": None})


def test_decompose_replaces_head_and_sets_num_labels():
    m = Toy()
    mb = decompose_binary(m, target_idx=2)  # pick class 2
    # Forward
    x = torch.randn(2, 4)
    out = mb(x=x)
    assert out.logits.shape[-1] == 2  # [target, negative]
    # Check that target row copied
    with torch.no_grad():
        w_pos = mb.classifier.pos.weight
        assert w_pos.shape == (1, 3)
