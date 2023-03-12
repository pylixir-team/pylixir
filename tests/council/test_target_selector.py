import pytest

from pylixir.core.base import Decision, GameState
from pylixir.council.base import CouncilTargetType, TargetSelector
from pylixir.council.target import NoneSelector, RandomSelector


@pytest.mark.parametrize("effect_index", [None, 1, 3, 4])
def test_none_selector(effect_index: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = NoneSelector(
        type=CouncilTargetType.none,
        condition=0,
        count=0,
    )
    assert [] == selector.select_targets(
        abundant_state, effect_index, any_random_number
    )


def test_random_selector(abundant_state: GameState) -> None:
    count = 1
    for random_number in range(10):
        selector = RandomSelector(
            type=CouncilTargetType.random,
            condition=0,
            count=count,
        )
        selected = selector.select_targets(abundant_state, None, random_number)
        assert count == len(selected)
