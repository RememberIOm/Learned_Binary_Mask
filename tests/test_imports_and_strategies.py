import types
from core.pruning.strategies import get_pruning_strategy


def test_factory_returns_strategies():
    assert type(get_pruning_strategy("wanda")).__name__ == "WandaStrategy"
    assert type(get_pruning_strategy("lbmask")).__name__ == "LBMaskStrategy"
    assert type(get_pruning_strategy("mi")).__name__ == "MutualInformationStrategy"
