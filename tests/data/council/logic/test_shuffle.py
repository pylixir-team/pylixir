import pytest

from pylixir.core.base import GameState
from pylixir.data.council.logic import ShuffleAll

# Each test will run this with random number(seed) 1~100


@pytest.mark.parametrize("locked_indices", [[], [1], [1, 2], [1, 2, 4], [3, 4]])
def test_shuffle(locked_indices: list[int], abundant_state: GameState) -> None:
    for idx in locked_indices:
        abundant_state.board.lock(idx)

    original_values = abundant_state.board.get_effect_values()
    total_run_count = 100
    logic = ShuffleAll(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    correct_count = 0
    equal_count = 0

    for random_number in range(total_run_count):
        changed_state = logic.reduce(abundant_state, [], random_number)
        changed_values = changed_state.board.get_effect_values()
        if set(changed_values) != set(original_values):
            continue
        if any(changed_values[idx] != original_values[idx] for idx in locked_indices):
            continue

        correct_count += 1
        if changed_values == original_values:
            equal_count += 1

    assert correct_count == total_run_count
    assert equal_count < correct_count
