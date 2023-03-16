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
    value: float
    remain_turn: int


class Effect(pydantic.BaseModel, metaclass=abc.ABCMeta):
    name: str
    value: int
    locked: bool
    max_value: int

    def lock(self) -> None:
        self.locked = True

    def unlock(self) -> None:
        self.locked = False

    def is_mutable(self) -> bool:
        return not self.locked and self.value < self.max_value


class GamePhase(enum.Enum):
    option = "option"
    council = "council"
    enchant = "enchant"
    done = "done"


MAX_EFFECT_COUNT = 11


class Board(pydantic.BaseModel):
    effects: tuple[Effect, Effect, Effect, Effect, Effect]

    def lock(self, effect_index: int) -> None:
        self.effects[effect_index].lock()

    def unlock(self, effect_index: int) -> None:
        self.effects[effect_index].unlock()

    def mutable_indices(self) -> list[int]:
        return [idx for idx, effect in enumerate(self.effects) if effect.is_mutable()]

    def get_effect_values(self) -> list[int]:
        return [effect.value for effect in self.effects]

    def modify_effect_count(self, effect_index: int, amount: int) -> None:
        basis = self.effects[effect_index].value
        basis += amount
        basis = min(max(0, basis), MAX_EFFECT_COUNT)
        self.effects[effect_index].value = basis

    def set_effect_count(self, effect_index: int, amount: int) -> None:
        self.effects[effect_index].value = amount

    def unlocked_indices(self) -> list[int]:
        return [idx for idx, effect in enumerate(self.effects) if not effect.locked]

    def locked_indices(self) -> list[int]:
        return [idx for idx, effect in enumerate(self.effects) if effect.locked]

    def get(self, idx: int) -> Effect:
        return self.effects[idx]

    def __len__(self) -> int:
        return len(self.effects)


class Enchanter(pydantic.BaseModel):
    _mutations: list[Mutation] = pydantic.PrivateAttr(default_factory=list)
    size: int = 5
    turn_left: int = 13

    def enchant(self, locked: list[int], random_number: float) -> list[int]:
        rng = RNG(random_number)
        random_numbers = [
            (rng.sample(), rng.sample()) for _ in range(self.get_enchant_effect_count())
        ]
        return self.expectable_enchant(
            self.query_enchant_prob(locked),
            self.query_lucky_ratio(),
            self.get_enchant_effect_count(),
            self.get_enchant_amount(),
            random_numbers,
        )

    def expectable_enchant(
        self,
        prob: list[float],
        lucky_ratio: list[float],
        count: int,
        amount: int,
        enchant_random_numbers: list[tuple[float, float]],
    ) -> list[int]:
        """명확한 랜덤 값을 사용하여 인챈트합니다.
        This is testable enchant function.
        """
        masked_prob = list(prob)
        result = [0 for _ in range(self.size)]
        assert count == len(enchant_random_numbers)

        for sampling_random_number, lucky_random_number in enchant_random_numbers:
            target_index = RNG.weighted_sampling(masked_prob, sampling_random_number)
            # add result as amount
            result[target_index] += amount
            if lucky_random_number < lucky_ratio[target_index]:
                result[target_index] += 1

            # pick and prevent duplicated sampling
            masked_prob[target_index] = 0

        return result

    def get_enchant_amount(self) -> int:
        for mutation in self._mutations:
            if mutation.target == MutationTarget.enchant_increase_amount:
                return int(mutation.value)

        return 1

    def get_enchant_effect_count(self) -> int:
        for mutation in self._mutations:
            if mutation.target == MutationTarget.enchant_effect_count:
                return int(mutation.value)

        return 1

    def query_enchant_prob(self, locked: list[int]) -> list[float]:
        available_slots = self.size - len(locked)
        distributed_prob = 1.0 / available_slots

        pick_ratios = [(0 if (idx in locked) else distributed_prob) for idx in range(5)]

        for mutation in self._mutations:
            if mutation.target != MutationTarget.prob:
                continue

            target_prob = pick_ratios[mutation.index]
            updated_prob = max(min(target_prob + mutation.value, 1.0), 0)
            actual_diff = updated_prob - target_prob

            for idx in range(5):
                if idx == mutation.index:
                    pick_ratios[idx] = updated_prob
                else:
                    if target_prob == 1:
                        pick_ratios[idx] == actual_diff  # pylint:disable=W0104
                    else:
                        pick_ratios[idx] = pick_ratios[idx] * (
                            1 - actual_diff / (1.0 - target_prob)
                        )

        return pick_ratios

    def query_lucky_ratio(self) -> list[float]:
        lucky_ratios = [0.1 for _ in range(5)]

        for mutation in self._mutations:
            if mutation.target != MutationTarget.lucky_ratio:
                continue

            lucky_ratios[mutation.index] = max(
                min(lucky_ratios[mutation.index] + mutation.value, 1), 0
            )
        return lucky_ratios

    def apply_mutation(self, mutation: Mutation) -> None:
        self._mutations.append(mutation)

    def mutate_prob(self, index: int, prob: float, remain_turn: int) -> None:
        self._mutations.append(
            Mutation(
                target=MutationTarget.prob,
                index=index,
                value=prob,
                remain_turn=remain_turn,
            )
        )

    def mutate_lucky_ratio(self, index: int, prob: float, remain_turn: int) -> None:
        self._mutations.append(
            Mutation(
                target=MutationTarget.lucky_ratio,
                index=index,
                value=prob,
                remain_turn=remain_turn,
            )
        )

    def increase_enchant_amount(self, value: int) -> None:
        self._mutations.append(
            Mutation(
                target=MutationTarget.enchant_increase_amount,
                index=-1,
                value=value,
                remain_turn=1,
            )
        )

    def change_enchant_effect_count(self, value: int) -> None:
        self._mutations.append(
            Mutation(
                target=MutationTarget.enchant_effect_count,
                index=-1,
                value=value,
                remain_turn=1,
            )
        )

    def consume_turn(self, count: int) -> None:
        self.turn_left -= count


class GameState(pydantic.BaseModel):
    phase: GamePhase
    reroll_left: int
    board: Board
    enchanter: Enchanter = pydantic.Field(default_factory=Enchanter)

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

        return self.enchanter.turn_left <= required_locks


class RNG:
    def __init__(self, start_seed: float):
        self._seed = start_seed

    def sample(self) -> float:
        sampled = self.chained_sample(self._seed)
        self._seed = sampled

        return sampled

    def fork(self) -> RNG:
        """
        fork to New RNG.
        fork will create new RNG, with modifiing self's seed.
        this is not idempotent; consequtive fork may yield different RNG.
        """
        forked_random_number_generator = RNG(self._seed + 1)
        self._seed += 1
        self.sample()

        return forked_random_number_generator

    @classmethod
    def chained_sample(cls, random_number: float) -> float:
        return Random(random_number).random()

    @classmethod
    def ranged(cls, min_range: int, max_range: int, random_number: float) -> int:
        bin_size = max_range - min_range + 1
        return int(random_number * bin_size) + min_range

    @classmethod
    def weighted_sampling(cls, probs: list[float], random_number: float) -> int:
        if sum(probs) == 0:
            raise ValueError("Summation of probability cannot be 0")

        pivot = random_number * sum(probs)
        cum_prob = 0.0

        for idx, prob in enumerate(probs):
            cum_prob += prob

            if cum_prob >= pivot and prob != 0:
                return idx

        raise ValueError("random_number cannot be 1")

    def shuffle(self, values: list[int]) -> list[int]:
        result = list(values)
        Random(self.sample()).shuffle(result)
        return result

    def pick(self, values: list[int]) -> int:
        shuffled = self.shuffle(values)
        return shuffled[0]
