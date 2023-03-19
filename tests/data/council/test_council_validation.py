import pytest

from pylixir.core.base import GameState
from pylixir.core.council import Council


@pytest.mark.parametrize(
    "turn_range, target_turn, expected",
    [
        ((0, 99), 3, True),
        ((1, 6), 0, False),
        ((1, 6), 3, True),
        ((1, 6), 6, True),
        ((1, 6), 1, True),
        ((1, 6), 11, False),
        ((9, 13), 5, False),
    ],
)
def test_turn_in_range(
    clean_state: GameState,
    turn_range: tuple[int, int],
    target_turn: int,
    expected: bool,
) -> None:
    council = Council(
        id="any",
        logics=[],
        pickup_ratio=10,
        turn_range=turn_range,
        slot_type=0,
        descriptions=[],
        type="common",
    )

    clean_state.consume_turn(target_turn)

    assert council.is_valid(clean_state) == expected
