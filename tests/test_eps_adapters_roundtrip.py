import pytest
from core.adversarial.adapters import ensure_eps_nlp, ensure_eps_vision


def test_nlp_accepts_strict_non_px_keys_only():
    a = ensure_eps_nlp({"start": 0.0, "max": 0.3, "step": 0.05})
    assert (a.start, a.max, a.step) == pytest.approx((0.0, 0.3, 0.05))
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start_px": 0.0, "max_px": 0.3, "step_px": 0.05})


def test_vision_accepts_strict_px_keys_only():
    a = ensure_eps_vision({"start_px": 0.0, "max_px": 8 / 255, "step_px": 2 / 255})
    assert a.step_px > 0
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start": 0.0, "max": 8 / 255, "step": 2 / 255})


def test_invalid_ranges_raise():
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start": 0.2, "max": 0.1, "step": 0.01})
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start_px": 0.1, "max_px": 0.05, "step_px": 0.0})
