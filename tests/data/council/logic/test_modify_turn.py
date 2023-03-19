import pytest

from pylixir.application.state import GameState
from pylixir.data.council.operation import DecreaseTurnLeft
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize("turn_count", [1, 2])
def test_decrease_turn(turn_count: int, abundant_state: GameState) -> None:
    operation = DecreaseTurnLeft(
        ratio=0,
        value=(turn_count, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )
    assert (
        changed_state.progress.get_turn_left()
        == abundant_state.progress.get_turn_left() - turn_count
    )
