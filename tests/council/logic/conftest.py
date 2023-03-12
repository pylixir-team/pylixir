import pytest

from pylixir.core.base import Effect, GamePhase, GameState, Sage


@pytest.fixture
def abundant_state() -> GameState:
    return GameState(
        phase=GamePhase.council,
        turn_left=10,
        reroll_left=1,
        effects=[
            Effect(name="A", value=7, locked=False),
            Effect(name="B", value=7, locked=False),
            Effect(name="C", value=5, locked=False),
            Effect(name="D", value=3, locked=False),
            Effect(name="E", value=3, locked=False),
        ],
        mutations=[],
        sages=[
            Sage(power=1, is_removed=False),
            Sage(power=1, is_removed=False),
            Sage(power=1, is_removed=False),
        ],
    )
