import abc
import enum
from typing import Optional

import pydantic

from pylixir.core.base import Randomness
from pylixir.core.state import GameState


class CouncilType(enum.Enum):
    lawfulLock = "lawfulLock"
    lawful = "lawful"
    chaosLock = "chaosLock"
    chaos = "chaos"
    lock = "lock"
    common = "common"
    exhausted = "exhausted"


class ElixirOperation(pydantic.BaseModel, metaclass=abc.ABCMeta):
    ratio: int
    value: tuple[int, int]
    remain_turn: int

    @abc.abstractmethod
    def reduce(
        self,
        state: GameState,
        targets: list[int],
        randomness: Randomness,
    ) -> GameState:
        ...

    @abc.abstractmethod
    def is_valid(self, state: GameState) -> bool:
        ...

    @classmethod
    def get_type(cls) -> str:
        class_name = cls.__name__
        return class_name[0].lower() + class_name[1:]


class TargetSelector(pydantic.BaseModel, metaclass=abc.ABCMeta):
    target_condition: int
    count: int

    @abc.abstractmethod
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        ...

    @abc.abstractmethod
    def is_valid(self, state: GameState) -> bool:
        ...


class Logic(pydantic.BaseModel):
    operation: ElixirOperation
    target_selector: TargetSelector

    def apply(
        self, state: GameState, effect_index: int, randomness: Randomness
    ) -> GameState:
        targets = self.target_selector.select_targets(state, effect_index, randomness)
        new_state = self.operation.reduce(state, targets, randomness)

        return new_state

    def is_valid(self, state: GameState) -> bool:
        return self.operation.is_valid(state) and self.target_selector.is_valid(state)


class Council(pydantic.BaseModel):
    id: str
    logics: list[Logic]
    pickup_ratio: int
    turn_range: tuple[int, int]
    slot_type: int
    descriptions: list[str]
    type: CouncilType

    def is_valid(self, state: GameState) -> bool:
        return self._is_turn_in_range(state) and all(
            logic.is_valid(state) for logic in self.logics
        )

    def _is_turn_in_range(self, state: GameState) -> bool:
        start, end = self.turn_range
        return start == 0 or start <= state.progress.get_current_turn() <= end


CouncilSet = tuple[Council, Council, Council]
