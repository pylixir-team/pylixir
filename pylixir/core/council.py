import abc
import enum
from typing import Optional

import pydantic

from pylixir.core.base import GameState


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


class CouncilTargetType(enum.Enum):
    none = "none"
    random = "random"
    proposed = "proposed"
    maxValue = "maxValue"
    minValue = "minValue"
    userSelect = "userSelect"
    lteValue = "lteValue"
    oneThreeFive = "oneThreeFive"
    twoFour = "twoFour"


class ElixirLogic(pydantic.BaseModel, metaclass=abc.ABCMeta):
    js_alias: str

    ratio: int
    value: tuple[int, int]
    remain_turn: int

    @abc.abstractmethod
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        ...


class TargetSelector(pydantic.BaseModel, metaclass=abc.ABCMeta):
    type: CouncilTargetType
    target_condition: int
    count: int

    @abc.abstractmethod
    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        ...


class Council(pydantic.BaseModel):
    id: str
    logic: ElixirLogic
    target_selector: TargetSelector


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
    def get_council(self, council_id: str) -> Council:
        ...

    def sample(self, state: GameState) -> tuple[str, str, str]:
        ...
