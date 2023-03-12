from __future__ import annotations

import abc
import enum
from random import Random
from typing import Optional

import pydantic


class Decision(pydantic.BaseModel):  # UIState
    sage_index: int
    effect_index: Optional[int]


class MutationTarget(enum.Enum):
    prob = "prob"
    lucky_ratio = "lucky_ratio"
    enchant_increase_amount = "enchant_increase_amount"
    enchant_effect_count = "enchant_effect_count"


class Mutation(pydantic.BaseModel):
    target: MutationTarget
    index: int
    value: int
    remain_turn: int


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


class Effect(pydantic.BaseModel, metaclass=abc.ABCMeta):
    name: str
    value: int
    locked: bool

    def lock(self) -> None:
        self.locked = True


class GamePhase(enum.Enum):
    option = "option"
    council = "council"
    enchant = "enchant"
    done = "done"


MAX_EFFECT_COUNT = 13


class GameState(pydantic.BaseModel):
    phase: GamePhase
    turn_left: int
    reroll_left: int
    effects: tuple[Effect, Effect, Effect, Effect, Effect]
    mutations: list[Mutation]
    sages: tuple[Sage, Sage, Sage]

    def add_mutation(self, mutation: Mutation) -> None:
        self.mutations.append(mutation)

    def enchant(self, random_number: float) -> None:
        ...

    def deepcopy(self) -> GameState:
        return self.copy(deep=True)

    def modify_effect_count(self, effect_index: int, amount: int) -> None:
        basis = self.effects[effect_index].value
        basis += amount
        basis = min(max(0, basis), MAX_EFFECT_COUNT)
        self.effects[effect_index].value = basis

    def set_effect_count(self, effect_index: int, amount: int) -> None:
        self.effects[effect_index].value = amount

    def consume_turn(self, count: int):
        self.turn_left -= count

    def get_effect_values(self) -> tuple[int, int, int, int, int]:
        return [effect.value for effect in self.effects]

    def lock(self, effect_index: int) -> None:
        self.effects[effect_index].lock()


class RNG:
    def sample(self) -> float:
        ...

    @classmethod
    def ranged(cls, min: int, max: int, random_number: float) -> int:
        bin_size = max - min + 1
        return int(random_number / 10000 * bin_size) + min

    @classmethod
    def shuffle(cls, values: list[int], random_number: float) -> list[int]:
        result = list(values)
        Random(random_number).shuffle(result)
        return result
