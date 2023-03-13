import pytest

from pylixir.core.base import GameState
from pylixir.data.council.logic import IncreaseReroll


@pytest.mark.parametrize("reroll_amount", [1, 2])
def test_increase_reroll(reroll_amount: int, abundant_state: GameState) -> None:
    logic = IncreaseReroll(
        ratio=0,
        value=(reroll_amount, 0),
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, [], 3456)

    assert changed_state.reroll_left == abundant_state.reroll_left + reroll_amount
