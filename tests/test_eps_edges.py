import pytest
from core.adversarial.adapters import ensure_eps_nlp, ensure_eps_vision


def test_nlp_invalid_raises():
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start": 0.3, "max": 0.2, "step": 0.01})
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start": 0.0, "max": 0.1, "step": 0.0})


def test_vision_invalid_raises():
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start": 0.2, "max": 0.1, "step": 0.01})
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start_px": 0.0, "max_px": 0.1, "step_px": 0.0})
