from pylixir.application.council import CouncilPool, Sage, SageCommittee
from pylixir.application.state import GameState
from pylixir.core.base import Board, Decision, Effect, GamePhase, Randomness


def state_initializer(max_value=10) -> GameState:
    return GameState(
        phase=GamePhase.council,
        reroll_left=2,
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
    )


def create_empty_committee(
    pool: CouncilPool, state: GameState, randomness: Randomness
) -> SageCommittee:
    sages = [Sage(power=0, is_removed=False, slot=idx) for idx in range(3)]

    return SageCommittee(
        sages=sages,
        councils=pool.get_council_set(state, sages, randomness, is_reroll=False),
    )
