from typing import Type

from pylixir.core.base import GameState, Randomness
from pylixir.core.council import ElixirOperation


class TargetSizeMismatchException(Exception):
    ...


class AlwaysValidOperation(ElixirOperation):
    def is_valid(self, state: GameState) -> bool:
        return True


class MutateProb(AlwaysValidOperation):
    def reduce(
        self,
        state: GameState,
        targets: list[int],
        randomness: Randomness,
    ) -> GameState:
        state = state.deepcopy()

        for index in targets:
            state.enchanter.mutate_prob(index, self.value[0], self.remain_turn)

        return state


class MutateLuckyRatio(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()

        for index in targets:
            state.enchanter.mutate_lucky_ratio(index, self.value[0], self.remain_turn)

        return state


class IncreaseTargetWithRatio(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()

        if randomness.binomial(self.ratio / 10000):
            state.board.modify_effect_count(target, self.value[0])

        return state


class IncreaseTargetRanged(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()

        diff_min, diff_max = self.value
        diff = randomness.uniform_int(diff_min, diff_max)

        state.board.modify_effect_count(target, diff)

        return state


class DecreaseTurnLeft(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
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
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()

        original_values = state.board.get_effect_values()
        unlocked_indices = state.board.unlocked_indices()
        locked_indices = state.board.locked_indices()

        starting = unlocked_indices + locked_indices

        shuffled_indices = randomness.shuffle(unlocked_indices)
        ending = shuffled_indices + locked_indices

        for start, end in zip(starting, ending):
            state.board.set_effect_count(start, original_values[end])

        return state


class SetEnchantTargetAndAmount(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
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
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()

        will_unlock = randomness.pick(state.board.locked_indices())
        will_lock = randomness.pick(state.board.unlocked_indices())

        state.board.lock(will_lock)
        state.board.unlock(will_unlock)

        return state

    def is_valid(self, state: GameState) -> bool:
        return len(state.board.locked_indices()) != 0


## TODO: implement this
class ChangeEffect(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state.deepcopy()


class LockTarget(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
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
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()
        state.modify_reroll(self.value[0])

        return state


class DecreasePrice(AlwaysValidOperation):
    """남은 모든 연성에서 비용이 <20%> 덜 들겠어."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state.deepcopy()


class Restart(AlwaysValidOperation):
    """이대론 안되겠어. 엘릭서의 효과와 단계를 <초기화>하겠다."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state.deepcopy()  ## TODO


class SetEnchantIncreaseAmount(AlwaysValidOperation):
    """이번에 연성되는 효과는 <2>단계 올라갈거야."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()
        state.enchanter.increase_enchant_amount(self.value[0])

        return state


class SetEnchantEffectCount(AlwaysValidOperation):
    """이번에는 <2>개의 효과를 동시에 연성하겠어"""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()
        state.enchanter.change_enchant_effect_count(self.value[0])

        return state

    def is_valid(self, state: GameState) -> bool:
        return state.requires_lock()


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
        DecreasePrice,
        Restart,
        SetEnchantIncreaseAmount,
        SetEnchantEffectCount,
    ]

    return operations
