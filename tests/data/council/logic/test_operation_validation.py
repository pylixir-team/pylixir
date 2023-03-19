import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import DecreaseTurnLeft


@pytest.mark.parametrize(
    "target_turn_left, turn_count, valid",
    [
        (10, 1, True),
        (3, 1, True),
        (2, 1, True),
        (1, 1, False),
        (0, 1, False),
        (4, 2, True),
        (3, 2, True),
        (2, 2, False),
    ],
)
def test_decrease_turn_requires_enough_turn_left(
    clean_state: GameState, target_turn_left: int, turn_count: int, valid: bool
) -> None:
    operation = DecreaseTurnLeft(
        ratio=0,
        value=(turn_count, 0),
        remain_turn=1,
    )
    clean_state.consume_turn(clean_state.turn_left - target_turn_left)

    assert operation.is_valid(clean_state) == valid
