import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import (
    IncreaseMaxAndDecreaseTarget,
    IncreaseMinAndDecreaseTarget,
    SetEnchantEffectCount,
    SetEnchantIncreaseAmount,
)
from tests.randomness import DeterministicRandomness


def test_set_enchant_increase_amount(abundant_state: GameState) -> None:
    enchant_amount = 2
    operation = SetEnchantIncreaseAmount(
        ratio=0, value=(enchant_amount, 0), remain_turn=1
    )

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )

    assert enchant_amount == changed_state.enchanter.get_enchant_amount()


def test_set_effect_enchant_count(abundant_state: GameState) -> None:
    enchant_count = 2
    operation = SetEnchantEffectCount(ratio=0, value=(enchant_count, 0), remain_turn=1)

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )

    assert enchant_count == changed_state.enchanter.get_enchant_effect_count()


@pytest.mark.parametrize(
    "targets, amount, start, end",
    [
        ([], (1, 0), [1, 3, 5, 7, 9], [1, 3, 5, 7, 10]),
        ([1], (1, -1), [1, 3, 5, 7, 9], [1, 2, 5, 7, 10]),
        ([2], (1, -1), [1, 3, 5, 7, 9], [1, 3, 4, 7, 10]),
        ([3], (1, -1), [1, 3, 5, 7, 9], [1, 3, 5, 6, 10]),
        ([4], (1, -1), [1, 3, 5, 7, 9], [1, 2, 5, 7, 10]),  # May subtle
    ],
)
def test_increase_max_and_decrease_target(
    targets: list[int],
    amount: tuple[int, int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
):
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    operation = IncreaseMaxAndDecreaseTarget(
        ratio=0,
        value=amount,
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, targets, DeterministicRandomness(0.123)
    )
    assert changed_state.board.get_effect_values() == end


@pytest.mark.parametrize(
    "targets, amount, start, end",
    [
        ([], (1, 0), [1, 3, 5, 7, 9], [2, 3, 5, 7, 9]),
        ([], (2, 0), [1, 3, 5, 7, 9], [3, 3, 5, 7, 9]),
        ([1], (1, -1), [1, 3, 5, 7, 9], [2, 2, 5, 7, 9]),
        ([2], (1, -1), [1, 3, 5, 7, 9], [2, 3, 4, 7, 9]),
        ([3], (1, -1), [1, 3, 5, 7, 9], [2, 3, 5, 6, 9]),
        ([0], (1, -1), [1, 3, 5, 7, 9], [2, 3, 4, 7, 9]),  # May subtle
    ],
)
def test_increase_min_and_decrease_target(
    targets: list[int],
    amount: tuple[int, int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
):
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    operation = IncreaseMinAndDecreaseTarget(
        ratio=0,
        value=amount,
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, targets, DeterministicRandomness(0.123)
    )
    assert changed_state.board.get_effect_values() == end
