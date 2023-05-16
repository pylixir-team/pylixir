from typing import Type, cast

from pylixir.application.council import (
    ElixirOperation,
    ForbiddenActionException,
    TargetSelector,
)
from pylixir.core.base import Randomness
from pylixir.core.state import GameState
from pylixir.data.council.common import (
    choose_max_indices,
    choose_min_indices,
    choose_random_indices_with_exclusion,
)
from pylixir.data.council.target import ProposedSelector


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
        enchanter = state.enchanter.copy()

        for index in targets:
            enchanter.mutate_prob(index, self.value[0] / 10000, self.remain_turn)

        state = state.copy(update=dict(enchanter=enchanter))
        return state


class MutateLuckyRatio(AlwaysValidOperation):
    """이번 연성에서 {0} 효과의 대성공 확률을 x% 올려주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        enchanter = state.enchanter.copy()

        for index in targets:
            enchanter.mutate_lucky_ratio(index, self.value[0] / 10000, self.remain_turn)

        state = state.copy(update=dict(enchanter=enchanter))

        return state


class IncreaseTargetWithRatio(AlwaysValidOperation):
    """<{0}> 효과의 단계를 <1> 올려보겠어. <25>% 확률로 성공하겠군."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        for target in targets:
            if randomness.binomial(self.ratio / 10000):
                board.modify_effect_count(target, self.value[0])

        state = state.copy(update=dict(board=board))

        return state


class IncreaseTargetRanged(AlwaysValidOperation):
    """<{0}> 효과의 단계를 [<+1>~<+2>]만큼 올려주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        board = state.board.copy()

        diff_min, diff_max = self.value
        diff = randomness.uniform_int(diff_min, diff_max)

        board.modify_effect_count(target, diff)

        state = state.copy(update=dict(board=board))

        return state


class DecreaseTurnLeft(ElixirOperation):
    """대신 기회를 2회 소모하겠군."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        progress = state.progress.copy()
        progress.spent_turn(self.consuming_turn)
        state = state.copy(update=dict(progress=progress))

        return state

    def is_valid(self, state: GameState) -> bool:
        return (
            state.progress.get_turn_left()
            > self.consuming_turn + len(state.board.locked_indices()) + 3
        )

    @property
    def consuming_turn(self) -> int:
        return self.value[0]


class ShuffleAll(AlwaysValidOperation):
    """<모든 효과>의 단계를 뒤섞도록 하지. 어떻게 뒤섞일지 보자고."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        original_values = board.get_effect_values()
        unlocked_indices = board.unlocked_indices()
        locked_indices = board.locked_indices()

        starting = unlocked_indices + locked_indices

        shuffled_indices = randomness.shuffle(unlocked_indices)
        ending = shuffled_indices + locked_indices

        for start, end in zip(starting, ending):
            board.set_effect_count(start, original_values[end])

        state = state.copy(update=dict(board=board))
        return state


class SetEnchantTargetAndAmount(AlwaysValidOperation):
    """이번에는 <{0}> 효과를 <2>단계 연성해주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]

        enchanter = state.enchanter.copy()

        enchanter.mutate_prob(target, 1.0, self.remain_turn)
        enchanter.increase_enchant_amount(self.value[0])

        state = state.copy(update=dict(enchanter=enchanter))

        return state


class UnlockAndLockOther(ElixirOperation):
    """<임의의 효과> <1>개의 봉인을 해제하고, 다른 효과 <1>개를 봉인해주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy(deep=True)

        will_unlock = randomness.pick(board.locked_indices())
        will_lock = randomness.pick(board.unlocked_indices())

        board.lock(will_lock)
        board.unlock(will_unlock)

        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        return len(state.board.locked_indices()) != 0


## TODO: implement this
class ChangeEffect(AlwaysValidOperation):
    """<네가 고르는> 슬롯의 효과를 바꿔주지. 어떤 효과일지 보자고."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state


class LockTarget(ElixirOperation):
    """<{0}> 효과를 봉인하겠다."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        board = state.board.copy()

        board.lock(target)
        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        return state.requires_lock()


class IncreaseReroll(AlwaysValidOperation):
    """조언이 더 필요한가? 다른 조언 보기 횟수를 <2>회 늘려주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        progress = state.progress.copy()

        progress.modify_reroll(self.value[0])
        state = state.copy(update=dict(progress=progress))

        return state


class DecreasePrice(AlwaysValidOperation):
    """남은 모든 연성에서 비용이 <20%> 덜 들겠어."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state


class Restart(AlwaysValidOperation):
    """이대론 안되겠어. 엘릭서의 효과와 단계를 <초기화>하겠다."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        raise ForbiddenActionException()  ## TODO


class SetEnchantIncreaseAmount(AlwaysValidOperation):
    """이번에 연성되는 효과는 <2>단계 올라갈거야."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        enchanter = state.enchanter.copy()

        enchanter.increase_enchant_amount(self.value[0])
        state = state.copy(update=dict(enchanter=enchanter))

        return state


class SetEnchantEffectCount(ElixirOperation):
    """이번에는 <2>개의 효과를 동시에 연성하겠어"""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        enchanter = state.enchanter.copy()

        enchanter.change_enchant_effect_count(self.value[0])
        state = state.copy(update=dict(enchanter=enchanter))

        return state

    def is_valid(self, state: GameState) -> bool:
        return len(state.board.mutable_indices()) >= self.value[0]


class SetValueRanged(AlwaysValidOperation):
    """<{0}> 효과의 단계를 [<1>~<2>] 중 하나로 바꿔주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        board = state.board.copy()

        value_min, value_max = self.value
        result = randomness.uniform_int(value_min, value_max)
        board.set_effect_count(target, result)
        state = state.copy(update=dict(board=board))

        return state

    def is_jointly_valid(
        self, state: GameState, target_selector: TargetSelector
    ) -> bool:
        """SetValueRanged Always runs with `proposed` targetType."""
        proposed_target_selector = cast(ProposedSelector, target_selector)
        target_index = proposed_target_selector.target_index
        return state.board.get_effect_values()[target_index] < self.value[1]


class RedistributeAll(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy(deep=True)

        unlocked_indices = board.unlocked_indices()

        candidates = [board.get(idx).value for idx in unlocked_indices]
        redistributed_values = randomness.redistribute(
            [0 for idx in range(len(candidates))],
            sum(candidates),
            board.get_max_value(),
        )

        for effect_idx, value in zip(unlocked_indices, redistributed_values):
            board.set_effect_count(effect_idx, value)

        state = state.copy(update=dict(board=board))
        return state


class RedistributeSelectedToOthers(AlwaysValidOperation):
    """<네가 고르는> 효과의 단계를 전부 다른 효과에 나누지. 어떻게 나뉠지 보자고."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]

        board = state.board.copy()

        redistribute_target_indices = [
            idx for idx in board.unlocked_indices() if idx != target
        ]

        candidates = [board.get(idx).value for idx in redistribute_target_indices]
        redistributed_values = randomness.redistribute(
            candidates,
            board.get_effect_values()[target],
            board.get_max_value(),
        )

        for effect_idx, value in zip(redistribute_target_indices, redistributed_values):
            board.set_effect_count(effect_idx, value)

        board.set_effect_count(target, 0)

        state = state.copy(update=dict(board=board))

        return state


class ShiftAll(AlwaysValidOperation):
    """
    <모든 효과>의 단계를 위로 <1> 슬롯 씩 옮겨주겠어.
    0=up, 1=down
    """

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        direction_offset = -1 if (self.value[0] == 0) else 1

        unlocked_indices = board.unlocked_indices()
        permuted_indices = [
            unlocked_indices[(idx + direction_offset) % len(unlocked_indices)]
            for idx in range(len(unlocked_indices))
        ]

        original_indices = unlocked_indices + board.locked_indices()
        target_indices = permuted_indices + board.locked_indices()

        original_values = board.get_effect_values()

        for original_index, target_index in zip(original_indices, target_indices):
            board.set_effect_count(target_index, original_values[original_index])

        state = state.copy(update=dict(board=board))

        return state


class SwapValues(ElixirOperation):
    """<{0}> 효과와 <{1}> 효과의 단계를 뒤바꿔줄게."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()
        original_values = board.get_effect_values()

        a_idx, b_idx = self.value

        board.set_effect_count(a_idx, original_values[b_idx])
        board.set_effect_count(b_idx, original_values[a_idx])

        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        return (state.progress.turn_passed > 0) and all(
            state.board.get(idx).is_mutable() for idx in self.value
        )


class SwapMinMax(ElixirOperation):
    """<최고 단계> 효과 <1>개와  <최하 단계> 효과 <1>개의 단계를 뒤바꿔주지."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        original_values = board.get_effect_values()
        choosed_min_index = choose_min_indices(board, randomness, count=1)[0]
        choosed_max_index = choose_max_indices(board, randomness, count=1)[0]

        board.set_effect_count(choosed_min_index, original_values[choosed_max_index])
        board.set_effect_count(choosed_max_index, original_values[choosed_min_index])

        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        return state.progress.turn_passed > 0


class Exhaust(ElixirOperation):
    """소진"""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        committee = state.committee.copy()
        state.committee.set_exhaust(self.value[0] - 1)

        state = state.copy(update=dict(committee=committee))

        return state

    def is_valid(self, state: GameState) -> bool:
        return len(state.committee.get_valid_slots()) == 3


class Exhausted(AlwaysValidOperation):
    """소진"""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        raise ForbiddenActionException()


class IncreaseMaxAndDecreaseTarget(AlwaysValidOperation):
    """<최고 단계> 효과 <1>개의 단계를 <1> 올려주지. 하지만 <최하 단계> 효과 <1>개의 단계는 <1> 내려갈 거야."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        original_values = board.get_effect_values()
        choosed_max_index = choose_max_indices(board, randomness, count=1)[0]

        max_value_increment, target_increment = self.value

        board.set_effect_count(
            choosed_max_index, original_values[choosed_max_index] + max_value_increment
        )

        for target_idx in targets:
            # prevents increase-decrease collision
            if target_idx == choosed_max_index:
                target_idx = choose_random_indices_with_exclusion(
                    board, randomness, excludes=[choosed_max_index]
                )

            board.set_effect_count(
                target_idx, original_values[target_idx] + target_increment
            )

        state = state.copy(update=dict(board=board))

        return state


class IncreaseMinAndDecreaseTarget(AlwaysValidOperation):
    """<최하 단계> 효과 <1>개의 단계를 <2> 올려주지. 하지만 <최고 단계> 효과 <1>개의 단계는 <2> 내려갈 거야."""

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        original_values = board.get_effect_values()
        choosed_min_index = choose_min_indices(board, randomness, count=1)[0]

        min_value_increment, target_increment = self.value

        board.set_effect_count(
            choosed_min_index, original_values[choosed_min_index] + min_value_increment
        )

        for target_idx in targets:
            # prevents increase-decrease collision
            if target_idx == choosed_min_index:
                target_idx = choose_random_indices_with_exclusion(
                    board, randomness, excludes=[choosed_min_index]
                )

            board.set_effect_count(
                target_idx, original_values[target_idx] + target_increment
            )

        state = state.copy(update=dict(board=board))

        return state


class RedistributeMinToOthers(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        choosed_min_index = choose_min_indices(board, randomness, count=1)[0]
        redistribute_target_indices = [
            idx for idx in board.unlocked_indices() if idx != choosed_min_index
        ]

        candidates = [board.get(idx).value for idx in redistribute_target_indices]
        redistributed_values = randomness.redistribute(
            candidates,
            board.get_effect_values()[choosed_min_index],
            board.get_max_value(),
        )

        for effect_idx, value in zip(redistribute_target_indices, redistributed_values):
            board.set_effect_count(effect_idx, value)

        board.set_effect_count(choosed_min_index, 0)

        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        return all(
            state.board.get(idx).value > 0 for idx in state.board.mutable_indices()
        )


class RedistributeMaxToOthers(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy(deep=True)

        choosed_max_index = choose_max_indices(board, randomness, count=1)[0]
        redistribute_target_indices = [
            idx for idx in board.unlocked_indices() if idx != choosed_max_index
        ]

        candidates = [board.get(idx).value for idx in redistribute_target_indices]
        redistributed_values = randomness.redistribute(
            candidates,
            board.get_effect_values()[choosed_max_index],
            board.get_max_value(),
        )

        for effect_idx, value in zip(redistribute_target_indices, redistributed_values):
            board.set_effect_count(effect_idx, value)

        board.set_effect_count(choosed_max_index, 0)

        state = state.copy(update=dict(board=board))

        return state


class DecreaseMaxAndSwapMinMax(AlwaysValidOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        original_values = board.get_effect_values()
        choosed_min_index = choose_min_indices(board, randomness, count=1)[0]
        choosed_max_index = choose_max_indices(board, randomness, count=1)[0]

        board.set_effect_count(
            choosed_min_index, original_values[choosed_max_index] - 1
        )
        board.set_effect_count(choosed_max_index, original_values[choosed_min_index])
        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        return state.progress.turn_passed > 0


class DecreaseFirstTargetAndSwap(ElixirOperation):
    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        board = state.board.copy()

        original_values = board.get_effect_values()
        first_target, second_target = self.value

        board.set_effect_count(second_target, original_values[first_target] - 1)
        board.set_effect_count(first_target, original_values[second_target])

        state = state.copy(update=dict(board=board))

        return state

    def is_valid(self, state: GameState) -> bool:
        if state.progress.turn_passed == 0:
            return False

        first_target, second_target = self.value
        return all(state.board.get(idx).is_mutable() for idx in self.value) and (
            state.board.get(first_target).value > state.board.get(second_target).value
        )


def get_operation_classes() -> list[Type[ElixirOperation]]:
    operations: list[Type[ElixirOperation]] = [
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
        SwapMinMax,
        SwapValues,
        Exhaust,
        IncreaseMaxAndDecreaseTarget,
        IncreaseMinAndDecreaseTarget,
        RedistributeMinToOthers,
        RedistributeMaxToOthers,
        DecreaseMaxAndSwapMinMax,
        DecreaseFirstTargetAndSwap,
    ]

    return operations
