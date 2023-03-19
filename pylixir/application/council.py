import abc
import enum
from typing import Optional

import pydantic

from pylixir.application.state import GameState
from pylixir.core.base import Decision, Randomness


class SageType(enum.Enum):
    none = "none"
    lawful = "lawful"
    chaos = "chaos"


class CouncilType(enum.Enum):
    lawfulLock = "lawfulLock"
    lawful = "lawful"
    chaosLock = "chaosLock"
    chaos = "chaos"
    lock = "lock"
    common = "common"
    exhausted = "exhausted"


MAX_LAWFUL = 3
MAX_CHAOS = -6


class Sage(pydantic.BaseModel):
    power: int
    is_removed: bool
    slot: int

    @property
    def type(self) -> SageType:
        if self.power == 0:
            return SageType.none

        if self.power > 0:
            return SageType.lawful

        return SageType.chaos

    def selected(self) -> None:
        if self.power < 0 or self.power == MAX_LAWFUL:
            self.power = 0

        self.power += 1

    def discarded(self) -> None:
        if self.power > 0 or self.power == MAX_CHAOS:
            self.power = 0

        self.power -= 1

    def is_lawful_max(self) -> bool:
        return self.power == MAX_LAWFUL

    def is_chaos_max(self) -> bool:
        return self.power == MAX_CHAOS


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
        self, state: GameState, decision: Decision, randomness: Randomness
    ) -> GameState:
        targets = self.target_selector.select_targets(
            state, decision.effect_index, randomness
        )
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
        return start <= state.progress.get_current_turn() <= end


CouncilSet = tuple[Council, Council, Council]


class SageCommittee(pydantic.BaseModel):
    sages: tuple[Sage, Sage, Sage]
    councils: CouncilSet

    def pick(self, picked_slot: int) -> None:
        for sage in self.sages:
            if sage.slot == picked_slot:
                sage.selected()
            else:
                sage.discarded()

    def get_council(self, sage_index: int) -> Council:
        return self.councils[sage_index]

    def set_councils(self, councils: CouncilSet) -> None:
        self.councils = councils


class CouncilPool:
    def __init__(
        self, councils: list[Council], trials_before_exact_sampling: int = 5
    ) -> None:
        self._councils = councils
        self._trials_before_exact_sampling = trials_before_exact_sampling

    def __len__(self) -> int:
        return len(self._councils)

    def get_council_set(
        self,
        state: GameState,
        sages: list[Sage],
        randomness: Randomness,
        is_reroll: bool = False,
    ) -> CouncilSet:
        # TODO: protection logic for reroll
        if is_reroll:
            pass

        return tuple([self.sample_council(state, sage, randomness) for sage in sages])

    def sample_council(
        self, state: GameState, sage: Sage, randomness: Randomness
    ) -> Council:
        council_type = self._get_council_type(state, sage)
        candidates = self.get_available_councils(sage.slot, council_type)

        weights = [float(council.pickup_ratio) for council in candidates]

        for _ in range(self._trials_before_exact_sampling):
            idx = randomness.weighted_sampling(weights)
            council = candidates[idx]
            if council.is_valid(state):
                return council

        refined_council = [council for council in candidates if council.is_valid(state)]

        refined_weights = [float(council.pickup_ratio) for council in refined_council]
        return refined_council[randomness.weighted_sampling(refined_weights)]

    def get_available_councils(
        self, sage_slot: int, council_type: CouncilType
    ) -> list[Council]:
        return [
            council
            for council in self._councils
            if council.type == council_type and (council.slot_type in (3, sage_slot))
        ]

    def _get_council_type(self, state: GameState, sage: Sage) -> CouncilType:
        if sage.is_removed:
            return CouncilType.exhausted

        if sage.is_lawful_max():
            if state.requires_lock():
                return CouncilType.lawfulLock

            return CouncilType.lawful

        if sage.is_chaos_max():
            if state.requires_lock():
                return CouncilType.chaosLock

            return CouncilType.chaos

        if state.requires_lock():
            return CouncilType.lock

        return CouncilType.common
