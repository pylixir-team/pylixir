from typing import Type

from pylixir.core.base import RNG, GameState
from pylixir.core.council import ElixirOperation


class TargetSizeMismatchException(Exception):
    ...


class AlwaysValidOperation(ElixirOperation):
    def is_valid(self, state: GameState) -> bool:
        return True


class MutateProb(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        for index in targets:
            state.enchanter.mutate_prob(index, self.value[0], self.remain_turn)

        return state


class MutateLuckyRatio(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        for index in targets:
            state.enchanter.mutate_lucky_ratio(index, self.value[0], self.remain_turn)

        return state


class IncreaseTargetWithRatio(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()

        if random_number <= self.ratio / 10000:
            state.board.modify_effect_count(target, self.value[0])

        return state


class IncreaseTargetRanged(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()

        diff_min, diff_max = self.value
        diff = RNG.ranged(diff_min, diff_max, random_number)

        state.board.modify_effect_count(target, diff)

        return state


class DecreaseTurnLeft(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()
        state.enchanter.consume_turn(self.consuming_turn)

        return state

    def is_valid(self, state: GameState) -> bool:
        return state.enchanter.turn_left >= self.consuming_turn + 1

    @property
    def consuming_turn(self) -> int:
        return self.value[0]


class ShuffleAll(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        original_values = state.board.get_effect_values()
        unlocked_indices = state.board.unlocked_indices()
        locked_indices = state.board.locked_indices()

        starting = unlocked_indices + locked_indices

        shuffled_indices = RNG(random_number).shuffle(unlocked_indices)
        ending = shuffled_indices + locked_indices

        for start, end in zip(starting, ending):
            state.board.set_effect_count(start, original_values[end])

        return state


class SetEnchantTargetAndAmount(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        state = state.deepcopy()
        target = targets[0]

        state.enchanter.mutate_prob(target, 1.0, self.remain_turn)
        state.enchanter.increase_enchant_amount(self.value[0])

        return state


class UnlockAndLockOther(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        rng = RNG(random_number)

        will_unlock = rng.fork().pick(state.board.locked_indices())
        will_lock = rng.fork().pick(state.board.unlocked_indices())

        state.board.lock(will_lock)
        state.board.unlock(will_unlock)

        return state

    def is_valid(self, state: GameState) -> bool:
        return len(state.board.locked_indices()) != 0


## TODO: implement this
class ChangeEffect(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        return state.deepcopy()


class LockTarget(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        state = state.deepcopy()
        target = targets[0]

        state.board.lock(target)

        return state

    def is_valid(self, state: GameState) -> bool:
        return state.requires_lock()


class IncreaseReroll(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()
        state.modify_reroll(self.value[0])

        return state


def get_operation_classes() -> list[Type[ElixirOperation]]:
    operations: list[Type[ElixirOperation]] = [
        AlwaysValidOperation,
        MutateProb,
        MutateLuckyRatio,
        IncreaseTargetWithRatio,
        IncreaseTargetRanged,
        DecreaseTurnLeft,
        ShuffleAll,
        SetEnchantTargetAndAmount,
        UnlockAndLockOther,
        ChangeEffect,
        LockTarget,
        IncreaseReroll,
    ]

    return operations
