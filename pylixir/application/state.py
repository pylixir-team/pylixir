from __future__ import annotations

import pydantic

from pylixir.core.base import Board, Enchanter, GamePhase

MAX_TURN_COUNT = 13


class GameState(pydantic.BaseModel):
    phase: GamePhase
    reroll_left: int
    board: Board
    enchanter: Enchanter = pydantic.Field(default_factory=Enchanter)
    turn_left: int = MAX_TURN_COUNT
    total_turn: int = MAX_TURN_COUNT

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def deepcopy(self) -> GameState:
        return self.copy(deep=True)

    def modify_reroll(self, amount: int) -> None:
        self.reroll_left += amount

    def requires_lock(self) -> bool:
        locked_effect_count = len(self.board.locked_indices())
        required_locks = 3 - locked_effect_count

        return self.turn_left <= required_locks

    def consume_turn(self, count: int) -> None:
        self.turn_left -= count

    def get_current_turn(self) -> int:
        return self.total_turn - self.turn_left
