import pytest

from pylixir.core.base import Board, Effect, GamePhase, GameState


@pytest.fixture
def clean_state() -> GameState:
    return GameState(
        phase=GamePhase.council,
        reroll_left=1,
        board=Board(
            effects=[
                Effect(name="A", value=0, locked=False, max_value=10),
                Effect(name="B", value=0, locked=False, max_value=10),
                Effect(name="C", value=0, locked=False, max_value=10),
                Effect(name="D", value=0, locked=False, max_value=10),
                Effect(name="E", value=0, locked=False, max_value=10),
            ],
            mutations=[],
        ),
    )


@pytest.fixture
def abundant_state() -> GameState:
    return GameState(
        phase=GamePhase.council,
        reroll_left=1,
        board=Board(
            effects=[
                Effect(name="A", value=7, locked=False, max_value=10),
                Effect(name="B", value=7, locked=False, max_value=10),
                Effect(name="C", value=5, locked=False, max_value=10),
                Effect(name="D", value=3, locked=False, max_value=10),
                Effect(name="E", value=3, locked=False, max_value=10),
            ],
            mutations=[],
        ),
    )


@pytest.fixture
def step_state() -> GameState:
    return GameState(
        phase=GamePhase.council,
        reroll_left=1,
        board=Board(
            effects=[
                Effect(name="A", value=1, locked=False, max_value=10),
                Effect(name="B", value=3, locked=False, max_value=10),
                Effect(name="C", value=5, locked=False, max_value=10),
                Effect(name="D", value=7, locked=False, max_value=10),
                Effect(name="E", value=9, locked=False, max_value=10),
            ],
            mutations=[],
        ),
    )
