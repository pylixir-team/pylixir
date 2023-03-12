import pytest

from pylixir.core.base import GameState
from pylixir.council.logic import (
    DecreaseTurnLeft
)
from tests.council.logic.util import assert_effect_changed


@pytest.mark.parametrize(
    "turn_count", [1,2]
)
def test_increase_target_with_ratio(
    turn_count: int, abundant_state: GameState
) -> None:
    logic = DecreaseTurnLeft(
        ratio=0,
        value=(turn_count, 0),
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, [], 2345)
    assert changed_state.turn_left == abundant_state.turn_left - turn_count
