import pytest

from pylixir.core.base import GameState
from pylixir.council.logic import LockTarget, UnlockAndLockOther


# Each test will run this with random number(seed) 1~100
@pytest.mark.parametrize("locked_indices", [[1], [1, 2], [1, 2, 4], [3, 4]])
def test_unlock_and_lock_other(
    locked_indices: list[int], abundant_state: GameState
) -> None:
    for idx in locked_indices:
        abundant_state.lock(idx)

    original_values = abundant_state.get_effect_values()
    total_run_count = 100

    logic = UnlockAndLockOther(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    correct_count = 0

    for random_number in range(total_run_count):
        changed_state = logic.reduce(abundant_state, [], random_number)
        changed_locks = [idx for idx in range(5) if changed_state.effects[idx].locked]

        if len(changed_locks) != len(locked_indices):  # check lock count preserved
            continue
        if (
            len(set(changed_locks) - set(locked_indices)) != 1
        ):  # check each diff is exactly equal to one
            continue
        if (
            len(set(locked_indices) - set(changed_locks)) != 1
        ):  # check each diff is exactly equal to one
            continue

        correct_count += 1

    assert correct_count == total_run_count


def test_lock_target(abundant_state: GameState) -> None:
    target_index = 3

    logic = LockTarget(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, [target_index], 3456)

    assert changed_state.effects[target_index].locked
