import pytest

from pylixir.application.council import ElixirOperation, ForbiddenActionException
from pylixir.core.state import GameState
from pylixir.data.council.operation import DecreasePrice, Restart
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize(
    "operation",
    [
        DecreasePrice(
            ratio=0,
            value=(0, 0),
            remain_turn=1,
        ),
    ],
)
def test_no_op(abundant_state: GameState, operation: ElixirOperation) -> None:
    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )

    assert abundant_state == changed_state


@pytest.mark.parametrize(
    "operation",
    [
        Restart(
            ratio=0,
            value=(0, 0),
            remain_turn=1,
        ),
    ],
)
def test_forbidden(abundant_state: GameState, operation: ElixirOperation) -> None:
    with pytest.raises(ForbiddenActionException):
        changed_state = operation.reduce(
            abundant_state, [], DeterministicRandomness(0.3456)
        )
