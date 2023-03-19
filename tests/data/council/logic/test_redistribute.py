import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import RedistributeAll
from tests.randomness import DeterministicRandomness

# Each test will run this with random number(seed) 1~100


@pytest.mark.parametrize("locked_indices", [[], [1], [1, 2], [1, 2, 4], [3, 4]])
def test_redistribute(locked_indices: list[int], abundant_state: GameState) -> None:
    for idx in locked_indices:
        abundant_state.board.lock(idx)

    operation = RedistributeAll(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    total_run_count = 100
    original_values = abundant_state.board.get_effect_values()

    correct_count = 0
    equal_count = 0

    for random_number in range(total_run_count):
        changed_state = operation.reduce(
            abundant_state, [], DeterministicRandomness(random_number)
        )
        changed_values = changed_state.board.get_effect_values()

        if sum(changed_values) != sum(original_values):
            continue
        if any(changed_values[idx] != original_values[idx] for idx in locked_indices):
            continue

        correct_count += 1
        if changed_values == original_values:
            equal_count += 1

    assert correct_count == total_run_count
    assert equal_count < correct_count
