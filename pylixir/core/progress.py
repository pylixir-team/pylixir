import enum

import pydantic

MAX_TURN_COUNT = 13


class GamePhase(enum.Enum):
    option = "option"
    council = "council"
    enchant = "enchant"
    done = "done"


class ProgressException(Exception):
    ...


class Progress(pydantic.BaseModel):
    turn_left: int = MAX_TURN_COUNT
    total_turn: int = MAX_TURN_COUNT
    reroll_left: int
    phase: GamePhase

    def get_turn_left(self) -> int:
        return self.turn_left

    def get_reroll_left(self) -> int:
        return self.reroll_left

    def spent_turn(self, count: int) -> None:
        self.turn_left -= count

    def get_current_turn(self) -> int:
        return self.total_turn - self.turn_left

    def modify_reroll(self, amount: int) -> None:
        self.reroll_left += amount

    def spent_reroll(self) -> None:
        if self.reroll_left <= 0:
            raise ProgressException("Reroll only available when reroll left")

        self.reroll_left -= 1

    @property
    def turn_passed(self) -> int:
        return self.total_turn - self.turn_left
