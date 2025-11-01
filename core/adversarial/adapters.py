# core/adversarial/adapters.py

from typing import Any, Dict, Union, Iterable
from core.adversarial.eps_cfg import EpsConfigNLP, EpsConfigVision


# -------- helpers (single responsibility: parse & validate dicts) --------
def _require_keys(d: Dict[str, Any], required: Iterable[str], ctx: str) -> None:
    """Ensure a dict strictly contains the required keys (at least) for the context."""
    missing = [k for k in required if k not in d]
    if missing:
        raise AssertionError(
            f"{ctx}: missing required keys {missing}. "
            f"Expected keys: {list(required)}"
        )


def _reject_keys(d: Dict[str, Any], forbidden: Iterable[str], ctx: str) -> None:
    """Reject dictionaries that include any forbidden keys to avoid silent misuse."""
    bad = [k for k in forbidden if k in d]
    if bad:
        raise AssertionError(
            f"{ctx}: keys {bad} are not allowed. "
            "Do not mix NLP/Vision formats. "
            "NLP: {start, max, step} | Vision: {start_px, max_px, step_px}"
        )


def ensure_eps_nlp(eps: Union[EpsConfigNLP, Dict[str, Any], None]) -> EpsConfigNLP:
    """Accept EpsConfigNLP or dict with STRICT keys {start, max, step}."""
    if eps is None or isinstance(eps, EpsConfigNLP):
        cfg = eps or EpsConfigNLP()
    else:
        ctx = "ensure_eps_nlp"
        _reject_keys(eps, ("start_px", "max_px", "step_px"), ctx)
        _require_keys(eps, ("start", "max", "step"), ctx)
        cfg = EpsConfigNLP(
            start=float(eps["start"]),
            max=float(eps["max"]),
            step=float(eps["step"]),
        )
    assert cfg.step > 0, "NLP eps: step must be > 0"
    assert cfg.start <= cfg.max, "NLP eps: start must be <= max"
    return cfg


def ensure_eps_vision(
    eps: Union[EpsConfigVision, Dict[str, Any], None],
) -> EpsConfigVision:
    """Accept EpsConfigVision or dict with STRICT keys {start_px, max_px, step_px}."""
    if eps is None or isinstance(eps, EpsConfigVision):
        cfg = eps or EpsConfigVision()
    else:
        ctx = "ensure_eps_vision"
        _reject_keys(eps, ("start", "max", "step"), ctx)
        _require_keys(eps, ("start_px", "max_px", "step_px"), ctx)
        cfg = EpsConfigVision(
            start_px=float(eps["start_px"]),
            max_px=float(eps["max_px"]),
            step_px=float(eps["step_px"]),
        )
    assert cfg.step_px > 0, "Vision eps: step_px must be > 0"
    assert cfg.start_px <= cfg.max_px, "Vision eps: start_px must be <= max_px"
    return cfg
