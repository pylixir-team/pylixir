import pytest

from pylixir.core.base import GameState
from pylixir.council.base import CouncilTargetType
from pylixir.council.target import (
    MaxValueSelector,
    MinValueSelector,
    NoneSelector,
    ProposedSelector,
    RandomSelector,
    UserSelector,
)


@pytest.mark.parametrize("effect_index", [None, 1, 3, 4])
def test_none_selector(effect_index: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = NoneSelector(
        type=CouncilTargetType.none,
        target_condition=0,
        count=0,
    )
    assert not selector.select_targets(abundant_state, effect_index, any_random_number)


def test_random_selector(abundant_state: GameState) -> None:
    count = 1
    for random_number in range(10):
        selector = RandomSelector(
            type=CouncilTargetType.random,
            target_condition=0,
            count=count,
        )
        selected = selector.select_targets(abundant_state, None, random_number)
        assert count == len(selected)


@pytest.mark.parametrize("target_condition", [1, 3, 4])
def test_proposed_selector(target_condition: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = ProposedSelector(
        type=CouncilTargetType.proposed,
        target_condition=target_condition,
        count=1,
    )

    selected = selector.select_targets(abundant_state, None, any_random_number)
    assert selected == [target_condition - 1]


def test_minimum_selector(abundant_state: GameState) -> None:
    for random_number in range(100):
        selector = MinValueSelector(
            type=CouncilTargetType.minValue,
            target_condition=0,
            count=1,
        )

        result = selector.select_targets(abundant_state, None, random_number)
        assert result in ([3], [4])


def test_maximum_selector(abundant_state: GameState) -> None:
    for random_number in range(100):
        selector = MaxValueSelector(
            type=CouncilTargetType.maxValue,
            target_condition=0,
            count=1,
        )

        result = selector.select_targets(abundant_state, None, random_number)
        assert result in ([0], [1])


@pytest.mark.parametrize("effect_index", [1, 3, 4])
def test_user_selector(effect_index: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = UserSelector(
        type=CouncilTargetType.none,
        target_condition=0,
        count=0,
    )
    assert selector.select_targets(abundant_state, effect_index, any_random_number) == [
        effect_index
    ]