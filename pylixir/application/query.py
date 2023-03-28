from pylixir.core.state import GameState
from pylixir.core.base import Board
from pylixir.core.committee import SageCommittee
from pylixir.core.progress import Progress
from pylixir.application.council import Council
from pylixir.application.service import CouncilPool


import pydantic



class GameStateView(pydantic.BaseModel):
    board: Board
    progress: Progress
    councils: tuple[Council, Council, Council]
    committee: SageCommittee

    def get_valid_sage_indices(self) -> list[int]:  # TODO: go to view
        return self.committee.get_valid_slots()

    def get_valid_effect_indices(self) -> list[int]:  # TODO: go to view
        return self.board.unlocked_indices()


def get_state_view(state: GameState, council_pool: CouncilPool) -> GameStateView:
    board = state.board.copy()
    progress = state.progress.copy()
    committee = state.committee.copy()
    councils = [
        council_pool.get_council(council_query) for council_query in state.suggestions
    ]

    return GameStateView(
        board=board,
        progress=progress,
        committee=committee,
        councils=councils
    )
