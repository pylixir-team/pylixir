import pytest

from pylixir.core.state import GameState
from pylixir.data.council.operation import DecreaseTurnLeft


@pytest.mark.parametrize(
    "target_turn_left, turn_count, valid",
    [
        (10, 1, True),
        (4, 1, False),
        (2, 1, False),
        (1, 1, False),
        (5, 1, True),
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
    clean_state.progress.spent_turn(
        clean_state.progress.get_turn_left() - target_turn_left
    )

    assert operation.is_valid(clean_state) == valid
