# pipelines/eps_cfg.py
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class EpsConfigNLP:
    """FGSM/PGD epsilon schedule for NLP models (embedding space, unit: L_inf in embedding scale)."""

    start: float = 0.0
    max: float = 0.25
    step: float = 0.01

    def as_tuple(self) -> Tuple[float, float, float]:
        """(start, max, step) for sweeps."""
        return (self.start, self.max, self.step)


@dataclass(frozen=True)
class EpsConfigVision:
    """FGSM/PGD epsilon schedule for Vision models (pixel scale, unit: raw pixel, e.g., 8/255)."""

    start_px: float = 0.0
    max_px: float = 8 / 255
    step_px: float = 2 / 255

    def as_tuple(self) -> Tuple[float, float, float]:
        """(start_px, max_px, step_px) for sweeps."""
        return (self.start_px, self.max_px, self.step_px)
