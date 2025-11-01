import pytest
from core.adversarial.adapters import ensure_eps_nlp, ensure_eps_vision


def test_nlp_eps_from_dict():
    cfg = ensure_eps_nlp({"start": 0.1, "max": 0.2, "step": 0.05})
    assert (cfg.start, cfg.max, cfg.step) == (0.1, 0.2, 0.05)


def test_vision_eps_accepts_strict_px_keys_only():
    a = ensure_eps_vision({"start_px": 0.0, "max_px": 8 / 255, "step_px": 2 / 255})
    assert a.start_px <= a.max_px and a.step_px > 0
