from __future__ import annotations

import pydantic

from pylixir.core.base import Board, Enchanter
from pylixir.core.progress import Progress

MAX_TURN_COUNT = 13


class GameState(pydantic.BaseModel):
    board: Board
    enchanter: Enchanter = pydantic.Field(default_factory=Enchanter)
    progress: Progress

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def deepcopy(self) -> GameState:
        return self.copy(deep=True)

    def requires_lock(self) -> bool:
        locked_effect_count = len(self.board.locked_indices())
        required_locks = 3 - locked_effect_count

        return self.progress.turn_left <= required_locks
