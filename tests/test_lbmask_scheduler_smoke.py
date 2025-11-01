import torch
from core.pruning.learned_binary_mask_pruning import (
    MaskPruneConfig,
    build_masked_model_from,
    ProgressivePruner,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Linear(8, 8)
        self.classifier = torch.nn.Linear(8, 2)
        self.config = type("C", (), {"num_labels": 2})()


def test_progressive_pruner_steps():
    m = Tiny()
    cfg = MaskPruneConfig(
        prune_during_train=True, end_sparsity=0.5, update_every=1, verbose=False
    )
    masked, overlays = build_masked_model_from(m, cfg, device=None)
    pruner = ProgressivePruner(
        masked,
        overlays,
        cfg,
        device=torch.device("cpu"),
        total_steps_hint=5,
        base_ref_model=m,
    )
    for s in range(5):
        pruner.on_step(s)  # should not raise
    assert True
