import pytest

from pylixir.core.state import GameState
from pylixir.data.council.operation import LockTarget, UnlockAndLockOther
from tests.randomness import DeterministicRandomness


# Each test will run this with random number(seed) 1~100
@pytest.mark.parametrize("locked_indices", [[1], [1, 2], [1, 2, 4], [3, 4]])
def test_unlock_and_lock_other(
    locked_indices: list[int], abundant_state: GameState
) -> None:
    for idx in locked_indices:
        abundant_state.board.lock(idx)

    total_run_count = 100

    operation = UnlockAndLockOther(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    correct_count = 0

    for random_number in range(total_run_count):
        changed_state = operation.reduce(
            abundant_state, [], DeterministicRandomness(random_number)
        )
        changed_locks = changed_state.board.locked_indices()

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

    operation = LockTarget(
        ratio=0,
        value=(0, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, [target_index], DeterministicRandomness(3456)
    )

    assert changed_state.board.get(target_index).locked
