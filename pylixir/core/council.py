import abc
import enum
from typing import Optional

import pydantic

from pylixir.core.base import RNG, Decision, GameState


class SageType(enum.Enum):
    none = "none"
    lawful = "lawful"
    chaos = "chaos"


class Sage(pydantic.BaseModel):
    power: int
    is_removed: bool

    @property
    def type(self) -> SageType:
        if self.power == 0:
            return SageType.none

        if self.power > 0:
            return SageType.lawful

        return SageType.chaos

    def run(self) -> None:
        ...

    def update_power(self, selected: bool) -> None:
        ...


class ElixirOperation(pydantic.BaseModel, metaclass=abc.ABCMeta):
    ratio: int
    value: tuple[int, int]
    remain_turn: int

    @abc.abstractmethod
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
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
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        ...

    @abc.abstractmethod
    def is_valid(self, state: GameState) -> bool:
        ...


class Logic(pydantic.BaseModel):
    operation: ElixirOperation
    target_selector: TargetSelector

    def apply(self, state: GameState, decision: Decision, rng: RNG) -> GameState:
        forked_rng = rng.fork()
        targets = self.target_selector.select_targets(
            state, decision.effect_index, forked_rng.sample()
        )
        forked_rng = rng.fork()
        new_state = self.operation.reduce(state, targets, forked_rng.sample())

        return new_state


class Council(pydantic.BaseModel):
    id: str
    logics: list[Logic]
    pickup_ratio: int
    turn_range: tuple[int, int]
    slot_type: int
    descriptions: list[str]
    type: str

    def is_valid(self) -> bool:
        ...

class SageCommittee(pydantic.BaseModel):
    sages: tuple[Sage, Sage, Sage]
    councils: tuple[Optional[Council], Optional[Council], Optional[Council]]

    def pick(self, sage_index: int) -> None:
        ...

    def get_council(self, sage_index: int) -> Council:
        maybe_council = self.councils[sage_index]
        if maybe_council is None:
            raise IndexError

        return maybe_council


class CouncilRepository:
    def __init__(self, councils) -> None:
        self._councils = councils

    def sample(self, state: GameState) -> tuple[Council, Council, Council]:
        ...

    def get_available_councils(self, sage_index: int) -> list[tuple[Council, int]]:
        ...


