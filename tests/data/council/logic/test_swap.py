import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import SwapValues
from tests.randomness import DeterministicRandomness

# Each test will run this with random number(seed) 1~100


@pytest.mark.parametrize(
    "swap_target, start, end",
    [
        ([0, 1], [1, 3, 5, 7, 9], [3, 1, 5, 7, 9]),
        ([0, 2], [1, 3, 5, 7, 9], [5, 3, 1, 7, 9]),
        ([1, 4], [1, 3, 5, 7, 9], [1, 9, 5, 7, 3]),
        ([2, 3], [1, 3, 5, 7, 9], [1, 3, 7, 5, 9]),
    ],
)
def test_redistribute(
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
