import pytest

from pylixir.core.base import GameState
from pylixir.data.council.common import choose_max_indices, choose_min_indices
from pylixir.data.council.operation import (
    RedistributeAll,
    RedistributeMaxToOthers,
    RedistributeMinToOthers,
    RedistributeSelectedToOthers,
)
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


@pytest.mark.parametrize("locked_indices", [[], [1], [1, 2], [1, 2, 4], [3, 4]])
def test_redistribute_selected_to_others(
    locked_indices: list[int], abundant_state: GameState
) -> None:
    for idx in locked_indices:
        abundant_state.board.lock(idx)

    operation = RedistributeSelectedToOthers(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    total_run_count, target = 100, 0
    original_values = abundant_state.board.get_effect_values()

    correct_count, equal_count = 0, 0

    for random_number in range(total_run_count):
        changed_state = operation.reduce(
            abundant_state, [target], DeterministicRandomness(random_number)
        )
        changed_values = changed_state.board.get_effect_values()

        if sum(changed_values) != sum(original_values):
            continue
        if any(changed_values[idx] != original_values[idx] for idx in locked_indices):
            continue
        if changed_values[target] != 0:
            continue

        correct_count += 1
        if changed_values == original_values:
            equal_count += 1

    assert correct_count == total_run_count
    assert equal_count < correct_count


@pytest.mark.parametrize("locked_indices", [[], [1], [1, 2], [1, 2, 4], [3, 4]])
def test_redistribute_min_to_others(
    locked_indices: list[int], step_state: GameState
) -> None:
    for idx in locked_indices:
        step_state.board.lock(idx)

    operation = RedistributeMinToOthers(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    total_run_count = 100
    original_values = step_state.board.get_effect_values()

    correct_count, equal_count = 0, 0

    for random_number in range(total_run_count):
        min_index = choose_min_indices(
            step_state.board, randomness=DeterministicRandomness(42), count=1
        )[
            0
        ]  # max index in step_state is always one
        changed_state = operation.reduce(
            step_state, [], DeterministicRandomness(random_number)
        )
        changed_values = changed_state.board.get_effect_values()

        if sum(changed_values) != sum(original_values):
            continue
        if any(changed_values[idx] != original_values[idx] for idx in locked_indices):
            continue
        if changed_values[min_index] != 0:
            continue

        correct_count += 1
        if changed_values == original_values:
            equal_count += 1

    assert correct_count == total_run_count
    assert equal_count < correct_count


def test_zero_value_forbidden_in_min_distribute(step_state: GameState):
    assert RedistributeMinToOthers(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    ).is_valid(step_state)

    step_state.board.set_effect_count(0, 0)
    assert not RedistributeMinToOthers(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    ).is_valid(step_state)


@pytest.mark.parametrize("locked_indices", [[], [1], [1, 2], [1, 2, 4], [3, 4]])
def test_redistribute_max_to_others(
    locked_indices: list[int], step_state: GameState
) -> None:
    for idx in locked_indices:
        step_state.board.lock(idx)

    operation = RedistributeMaxToOthers(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    total_run_count = 100
    original_values = step_state.board.get_effect_values()

    correct_count, equal_count = 0, 0

    for random_number in range(total_run_count):
        max_index = choose_max_indices(
            step_state.board, randomness=DeterministicRandomness(42), count=1
        )[
            0
        ]  # min index in step_state is always one
        changed_state = operation.reduce(
            step_state, [], DeterministicRandomness(random_number)
        )
        changed_values = changed_state.board.get_effect_values()

        if sum(changed_values) != sum(original_values):
            continue
        if any(changed_values[idx] != original_values[idx] for idx in locked_indices):
            continue
        if changed_values[max_index] != 0:
            continue

        correct_count += 1
        if changed_values == original_values:
            equal_count += 1

    assert correct_count == total_run_count
    assert equal_count < correct_count
