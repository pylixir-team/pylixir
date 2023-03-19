from typing import Type

from pylixir.core.base import GameState, Randomness
from pylixir.core.council import ElixirOperation


class TargetSizeMismatchException(Exception):
    ...


class AlwaysValidOperation(ElixirOperation):
    def is_valid(self, state: GameState) -> bool:
        return True


class MutateProb(AlwaysValidOperation):
    """이번 연성에서 {0} 효과가 연성될 확률을 x% 올려주지."""

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
    """이번 연성에서 {0} 효과의 대성공 확률을 x% 올려주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()

        for index in targets:
            state.enchanter.mutate_lucky_ratio(index, self.value[0], self.remain_turn)

        return state


class IncreaseTargetWithRatio(AlwaysValidOperation):
    """<{0}> 효과의 단계를 <1> 올려보겠어. <25>% 확률로 성공하겠군."""

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
    """<{0}> 효과의 단계를 [<+1>~<+2>]만큼 올려주지."""

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
    """대신 기회를 2회 소모하겠군."""

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
    """<모든 효과>의 단계를 뒤섞도록 하지. 어떻게 뒤섞일지 보자고."""

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
    """이번에는 <{0}> 효과를 <2>단계 연성해주지."""

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
    """<임의의 효과> <1>개의 봉인을 해제하고, 다른 효과 <1>개를 봉인해주지."""

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
    """<네가 고르는> 슬롯의 효과를 바꿔주지. 어떤 효과일지 보자고."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state.deepcopy()


class LockTarget(ElixirOperation):
    """<{0}> 효과를 봉인하겠다."""

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
    """조언이 더 필요한가? 다른 조언 보기 횟수를 <2>회 늘려주지."""

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


class SetValueRanged(AlwaysValidOperation):
    """<{0}> 효과의 단계를 [<1>~<2>] 중 하나로 바꿔주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()
        value_min, value_max = self.value
        result = randomness.uniform_int(value_min, value_max)
        state.board.set_effect_count(target, result)

        return state


class RedistributeAll(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()

        unlocked_indices = state.board.unlocked_indices()

        candidates = [state.board.get(idx).value for idx in unlocked_indices]
        redistributed_values = randomness.redistribute(
            [0 for idx in range(len(candidates))],
            sum(candidates),
            state.board.get_max_value(),
        )

        for effect_idx, value in zip(unlocked_indices, redistributed_values):
            state.board.set_effect_count(effect_idx, value)

        return state


class RedistributeSelectedToOthers(AlwaysValidOperation):
    """<네가 고르는> 효과의 단계를 전부 다른 효과에 나누지. 어떻게 나뉠지 보자고."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]

        state = state.deepcopy()

        redistribute_target_indices = [
            idx for idx in state.board.unlocked_indices() if idx != target
        ]

        candidates = [state.board.get(idx).value for idx in redistribute_target_indices]
        redistributed_values = randomness.redistribute(
            candidates,
            state.board.get_effect_values()[target],
            state.board.get_max_value(),
        )

        for effect_idx, value in zip(redistribute_target_indices, redistributed_values):
            state.board.set_effect_count(effect_idx, value)

        state.board.set_effect_count(target, 0)

        return state


class ShiftAll(AlwaysValidOperation):
    """
    <모든 효과>의 단계를 위로 <1> 슬롯 씩 옮겨주겠어.
    0=up, 1=down
    """

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        state = state.deepcopy()

        direction_offset = -1 if (self.value[0] == 0) else 1

        unlocked_indices = state.board.unlocked_indices()
        permuted_indices = [
            unlocked_indices[(idx + direction_offset) % len(unlocked_indices)]
            for idx in range(len(unlocked_indices))
        ]

        original_indices = unlocked_indices + state.board.locked_indices()
        target_indices = permuted_indices + state.board.locked_indices()

        original_values = state.board.get_effect_values()

        for original_index, target_index in zip(original_indices, target_indices):
            state.board.set_effect_count(target_index, original_values[original_index])

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
        DecreasePrice,
        Restart,
        SetEnchantIncreaseAmount,
        SetEnchantEffectCount,
        SetValueRanged,
        RedistributeAll,
        RedistributeSelectedToOthers,
        ShiftAll,
    ]

    return operations
