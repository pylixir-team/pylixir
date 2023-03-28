from __future__ import annotations

from typing import Callable

import pydantic

from pylixir.core.base import Board, Enchanter, Randomness
from pylixir.core.committee import SageCommittee
from pylixir.core.progress import Progress

MAX_TURN_COUNT = 13


class CouncilQuery(pydantic.BaseModel):
    id: str


class GameState(pydantic.BaseModel):
    board: Board
    enchanter: Enchanter = pydantic.Field(default_factory=Enchanter)
    progress: Progress
    suggestions: tuple[CouncilQuery, CouncilQuery, CouncilQuery]
    committee: SageCommittee

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def deepcopy(self, **kwargs: dict[str, bool]) -> GameState:
        update = {
            target: getattr(self, target).copy(deep=True)
            for target, value in kwargs.items()
            if value
        }
        return self.copy(update=update)

    def requires_lock(self) -> bool:
        locked_effect_count = len(self.board.locked_indices())
        required_locks = 3 - locked_effect_count

        return self.progress.turn_left <= required_locks


Reducer = Callable[[GameState, Randomness], GameState]
