import pytest
from core.adversarial.adapters import ensure_eps_nlp, ensure_eps_vision
from core.adversarial.eps_cfg import EpsConfigNLP, EpsConfigVision

# --- happy paths -------------------------------------------------------------


def test_nlp_accepts_strict_non_px_keys():
    cfg = ensure_eps_nlp({"start": 0.05, "max": 0.15, "step": 0.02})
    assert (cfg.start, cfg.max, cfg.step) == pytest.approx((0.05, 0.15, 0.02))


def test_vision_accepts_strict_px_keys():
    cfg = ensure_eps_vision(
        {"start_px": 1 / 255, "max_px": 7 / 255, "step_px": 2 / 255}
    )
    assert (cfg.start_px, cfg.max_px, cfg.step_px) == pytest.approx(
        (1 / 255, 7 / 255, 2 / 255)
    )


def test_defaults_when_none():
    assert isinstance(ensure_eps_nlp(None), EpsConfigNLP)
    assert isinstance(ensure_eps_vision(None), EpsConfigVision)


# --- rejection paths (no mixing) --------------------------------------------


def test_nlp_rejects_px_keys():
    with pytest.raises(AssertionError) as e:
        ensure_eps_nlp({"start_px": 0.01, "max_px": 0.02, "step_px": 0.001})
    assert "not allowed" in str(e.value)


def test_vision_rejects_non_px_keys():
    with pytest.raises(AssertionError) as e:
        ensure_eps_vision({"start": 0.01, "max": 0.02, "step": 0.001})
    assert "not allowed" in str(e.value)


# --- boundary checks ---------------------------------------------------------


def test_nlp_invalid_ranges_raise():
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start": 0.2, "max": 0.1, "step": 0.01})
    with pytest.raises(AssertionError):
        ensure_eps_nlp({"start": 0.0, "max": 0.1, "step": 0.0})


def test_vision_invalid_ranges_raise():
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start_px": 0.02, "max_px": 0.01, "step_px": 0.001})
    with pytest.raises(AssertionError):
        ensure_eps_vision({"start_px": 0.0, "max_px": 0.1, "step_px": 0.0})
