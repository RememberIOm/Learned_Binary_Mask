import pytest
from core.adversarial.adapters import ensure_eps_nlp, ensure_eps_vision


def test_nlp_values_from_aliased_keys_are_used():
    # Strict non-px keys
    cfg = ensure_eps_nlp({"start": 0.07, "max": 0.13, "step": 0.02})
    assert (cfg.start, cfg.max, cfg.step) == pytest.approx((0.07, 0.13, 0.02))
    # Mixed style should be rejected
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start_px": 0.07, "max_px": 0.13, "step_px": 0.02})


def test_vision_values_from_cli_style_are_used():
    # Strict px keys
    cfg = ensure_eps_vision({"start_px": 0.011, "max_px": 0.019, "step_px": 0.003})
    assert cfg.start_px == pytest.approx(0.011)
    assert cfg.max_px == pytest.approx(0.019)
    assert cfg.step_px == pytest.approx(0.003)
    # Non-px style should be rejected
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start": 0.011, "max": 0.019, "step": 0.003})
