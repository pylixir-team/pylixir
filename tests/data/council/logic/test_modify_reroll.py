import pytest

from pylixir.application.state import GameState
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

    assert (
        changed_state.progress.get_reroll_left()
        == abundant_state.progress.get_reroll_left() + reroll_amount
    )
