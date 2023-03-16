import pytest

from pylixir.core.base import GameState
from pylixir.data.council.operation import IncreaseReroll
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize("reroll_amount", [1, 2])
def test_increase_reroll(reroll_amount: int, abundant_state: GameState) -> None:
    operation = IncreaseReroll(
        ratio=0,
        value=(reroll_amount, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )

    assert changed_state.reroll_left == abundant_state.reroll_left + reroll_amount
