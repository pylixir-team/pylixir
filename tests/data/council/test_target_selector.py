import pytest

from pylixir.core.base import GameState
from pylixir.data.council.target import (
    LteValueSelector,
    MaxValueSelector,
    MinValueSelector,
    NoneSelector,
    OneThreeFiveSelector,
    ProposedSelector,
    RandomSelector,
    TwoFourSelector,
    UserSelector,
)


@pytest.mark.parametrize("effect_index", [None, 1, 3, 4])
def test_none_selector(effect_index: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = NoneSelector(
        target_condition=0,
        count=0,
    )
    assert not selector.select_targets(abundant_state, effect_index, any_random_number)


def test_random_selector(abundant_state: GameState) -> None:
    count = 1
    for random_number in range(10):
        selector = RandomSelector(
            target_condition=0,
            count=count,
        )
        selected = selector.select_targets(abundant_state, None, random_number)
        assert count == len(selected)


@pytest.mark.parametrize("target_condition", [1, 3, 4])
def test_proposed_selector(target_condition: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = ProposedSelector(
        target_condition=target_condition,
        count=1,
    )

    selected = selector.select_targets(abundant_state, None, any_random_number)
    assert selected == [target_condition - 1]


def test_minimum_selector(abundant_state: GameState) -> None:
    for random_number in range(100):
        selector = MinValueSelector(
            target_condition=0,
            count=1,
        )

        result = selector.select_targets(abundant_state, None, random_number)
        assert result in ([3], [4])


def test_maximum_selector(abundant_state: GameState) -> None:
    for random_number in range(100):
        selector = MaxValueSelector(
            target_condition=0,
            count=1,
        )

        result = selector.select_targets(abundant_state, None, random_number)
        assert result in ([0], [1])


@pytest.mark.parametrize("effect_index", [1, 3, 4])
def test_user_selector(effect_index: int, abundant_state: GameState) -> None:
    any_random_number = 42
    selector = UserSelector(
        target_condition=0,
        count=0,
    )
    assert selector.select_targets(abundant_state, effect_index, any_random_number) == [
        effect_index
    ]


@pytest.mark.parametrize(
    "target_condition, expected",
    [(0, []), (1, []), (3, [3, 4]), (5, [2, 3, 4]), (6, [2, 3, 4])],
)
def test_lte_selector(
    target_condition: int, expected: list[int], abundant_state: GameState
) -> None:
    any_random_number = 42
    selector = LteValueSelector(
        target_condition=target_condition,
        count=1,
    )

    selected = selector.select_targets(abundant_state, None, any_random_number)
    assert expected == selected


def test_one_three_five(abundant_state: GameState) -> None:
    any_random_number = 42
    selector = OneThreeFiveSelector(
        target_condition=0,
        count=0,
    )
    assert selector.select_targets(abundant_state, None, any_random_number) == [0, 2, 4]


def test_two_four(abundant_state: GameState) -> None:
    any_random_number = 42
    selector = TwoFourSelector(
        target_condition=0,
        count=0,
    )
    assert selector.select_targets(abundant_state, None, any_random_number) == [1, 3]
