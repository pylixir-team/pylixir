from typing import Type

import pytest

from pylixir.application.council import ElixirOperation
from pylixir.core.state import GameState
from pylixir.data.council.operation import (
    DecreaseFirstTargetAndSwap,
    DecreaseMaxAndSwapMinMax,
    SwapMinMax,
    SwapValues,
)
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize(
    "swap_target, start, end",
    [
        ([0, 1], [1, 3, 5, 7, 9], [3, 1, 5, 7, 9]),
        ([0, 2], [1, 3, 5, 7, 9], [5, 3, 1, 7, 9]),
        ([1, 4], [1, 3, 5, 7, 9], [1, 9, 5, 7, 3]),
        ([2, 3], [1, 3, 5, 7, 9], [1, 3, 7, 5, 9]),
    ],
)
def test_swap_between_values(
    swap_target: tuple[int, int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    operation = SwapValues(
        ratio=0,
        value=swap_target,
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end


@pytest.mark.parametrize(
    "locked_indices, start, end",
    [
        ([], [1, 3, 5, 7, 9], [9, 3, 5, 7, 1]),
        ([1], [1, 3, 5, 7, 9], [9, 3, 5, 7, 1]),
        ([1, 2], [1, 3, 5, 7, 9], [9, 3, 5, 7, 1]),
        ([0], [1, 3, 5, 7, 9], [1, 9, 5, 7, 3]),
        ([0, 1, 2], [1, 3, 5, 7, 9], [1, 3, 5, 9, 7]),
    ],
)
def test_swap_min_max(
    locked_indices: list[int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    for idx in locked_indices:
        abundant_state.board.lock(idx)

    operation = SwapMinMax(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end


@pytest.mark.parametrize(
    "locked_indices, start, end",
    [
        ([], [1, 3, 5, 7, 9], [8, 3, 5, 7, 1]),
        ([1], [1, 3, 5, 7, 9], [8, 3, 5, 7, 1]),
        ([1, 2], [1, 3, 5, 7, 9], [8, 3, 5, 7, 1]),
        ([0], [1, 3, 5, 7, 9], [1, 8, 5, 7, 3]),
        ([0, 1, 2], [1, 3, 5, 7, 9], [1, 3, 5, 8, 7]),
        ([4], [1, 3, 5, 7, 9], [6, 3, 5, 1, 9]),
    ],
)
def test_decrease_max_and_swap_min_max(
    locked_indices: list[int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    for idx in locked_indices:
        abundant_state.board.lock(idx)

    operation = DecreaseMaxAndSwapMinMax(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end


@pytest.mark.parametrize(
    "targets, start, end",
    [
        ([1, 0], [1, 3, 5, 7, 9], [2, 1, 5, 7, 9]),
        ([3, 0], [1, 3, 5, 7, 9], [6, 3, 5, 1, 9]),
        ([2, 1], [1, 3, 5, 7, 9], [1, 4, 3, 7, 9]),
        ([4, 3], [1, 3, 5, 7, 9], [1, 3, 5, 8, 7]),
    ],
)
def test_decrease_first_and_swap(
    targets: tuple[int, int],
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    operation = DecreaseFirstTargetAndSwap(
        ratio=0,
        value=targets,
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end


@pytest.mark.parametrize(
    "targets, start",
    [
        ([1, 0], [1, 3, 5, 7, 9]),
        ([3, 0], [1, 3, 5, 7, 9]),
        ([2, 1], [1, 3, 5, 7, 9]),
        ([4, 3], [1, 3, 5, 7, 9]),
    ],
)
def testtest_decrease_first_and_swap_is_not_valid_in_reverse_order(
    targets: tuple[int, int],
    start: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])
        abundant_state.progress.turn_left -= 1

    assert DecreaseFirstTargetAndSwap(
        ratio=0,
        value=targets,
        remain_turn=1,
    ).is_valid(abundant_state)

    assert not DecreaseFirstTargetAndSwap(
        ratio=0,
        value=list(reversed(targets)),
        remain_turn=1,
    ).is_valid(abundant_state)


@pytest.mark.parametrize(
    "swap_opration",
    [
        SwapValues,
        SwapMinMax,
        DecreaseMaxAndSwapMinMax,
        DecreaseFirstTargetAndSwap,
    ],
)
def test_starting_state_not_valid(
    swap_opration: Type[ElixirOperation],
    abundant_state: GameState,
) -> None:
    for idx, count in zip(range(5), [1, 3, 5, 7, 9]):
        abundant_state.board.set_effect_count(idx, count)

    abundant_state.progress.turn_left = abundant_state.progress.total_turn

    assert not swap_opration(
        ratio=0,
        value=[1, 0],
        remain_turn=1,
    ).is_valid(abundant_state)
