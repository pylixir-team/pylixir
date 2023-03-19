import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import ShiftAll
from tests.randomness import DeterministicRandomness

# Each test will run this with random number(seed) 1~100


@pytest.mark.parametrize(
    "locked_indices, direction, start, end",
    [
        ([], 0, [1, 3, 5, 7, 9], [3, 5, 7, 9, 1]),
        ([], 1, [1, 3, 5, 7, 9], [9, 1, 3, 5, 7]),
        ([1], 0, [1, 3, 5, 7, 9], [5, 3, 7, 9, 1]),
        ([1], 1, [1, 3, 5, 7, 9], [9, 3, 1, 5, 7]),
        ([1, 3], 0, [1, 3, 5, 7, 9], [5, 3, 9, 7, 1]),
        ([1, 3], 1, [1, 3, 5, 7, 9], [9, 3, 1, 7, 5]),
        ([0, 2, 4], 0, [1, 3, 5, 7, 9], [1, 7, 5, 3, 9]),
        ([0, 1, 2], 1, [1, 3, 5, 7, 9], [1, 3, 5, 9, 7]),
    ],
)
def test_redistribute(
    locked_indices: list[int],
    direction: int,
    start: list[int],
    end: list[int],
    abundant_state: GameState,
) -> None:
    for idx in range(5):
        abundant_state.board.set_effect_count(idx, start[idx])

    for idx in locked_indices:
        abundant_state.board.lock(idx)

    operation = ShiftAll(
        ratio=0,
        value=(direction, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(abundant_state, [], DeterministicRandomness(42))
    assert changed_state.board.get_effect_values() == end
