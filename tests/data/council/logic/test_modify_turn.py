import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import DecreaseTurnLeft
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize("turn_count", [1, 2])
def test_increase_target_with_ratio(turn_count: int, abundant_state: GameState) -> None:
    operation = DecreaseTurnLeft(
        ratio=0,
        value=(turn_count, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )
    assert (
        changed_state.enchanter.turn_left
        == abundant_state.enchanter.turn_left - turn_count
    )
