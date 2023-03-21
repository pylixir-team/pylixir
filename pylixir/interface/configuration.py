from pylixir.core.base import Board, Effect
from pylixir.core.committee import Sage, SageCommittee
from pylixir.core.progress import GamePhase, Progress
from pylixir.core.state import CouncilQuery, GameState


def create_empty_committee() -> SageCommittee:
    sages = [Sage(power=0, is_removed=False, slot=idx) for idx in range(3)]

    return SageCommittee(
        sages=sages,
    )


def state_initializer(max_value: int = 10) -> GameState:
    return GameState(
        progress=Progress(
            phase=GamePhase.council,
            reroll_left=2,
        ),
        board=Board(
            effects=[
                Effect(name="A", value=0, locked=False, max_value=max_value),
                Effect(name="B", value=0, locked=False, max_value=max_value),
                Effect(name="C", value=0, locked=False, max_value=max_value),
                Effect(name="D", value=0, locked=False, max_value=max_value),
                Effect(name="E", value=0, locked=False, max_value=max_value),
            ],
            mutations=[],
        ),
        committee=create_empty_committee(),
        suggestions=[CouncilQuery(id=""), CouncilQuery(id=""), CouncilQuery(id="")],
    )
