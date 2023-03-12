from pylixir.core.base import RNG, GameState, Mutation, MutationTarget
from pylixir.council.base import ElixirLogic


class TargetSizeMismatchException(Exception):
    ...


class MutateProb(ElixirLogic):
    js_alias: str = "mutateProb"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        mutations = [
            Mutation(
                target=MutationTarget.prob,
                index=index,
                value=self.value[0],
                remain_turn=self.remain_turn,
            )
            for index in targets
        ]

        state.mutations += mutations
        return state


class MutateLuckyRatio(ElixirLogic):
    js_alias: str = "mutateLuckyRatio"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        mutations = [
            Mutation(
                target=MutationTarget.lucky_ratio,
                index=index,
                value=self.value[0],
                remain_turn=self.remain_turn,
            )
            for index in targets
        ]

        state.mutations += mutations
        return state


class IncreaseTargetWithRatio(ElixirLogic):
    js_alias: str = "increaseTargetWithRatio"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()

        if random_number <= self.ratio:
            state.modify_effect_count(target, self.value[0])

        return state


class IncreaseTargetRanged(ElixirLogic):
    js_alias: str = "increaseTargetRanged"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        target = targets[0]
        state = state.deepcopy()

        diff_min, diff_max = self.value
        diff = RNG.ranged(diff_min, diff_max, random_number)

        state.modify_effect_count(target, diff)

        return state


class DecreaseTurnLeft(ElixirLogic):
    js_alias: str = "decreaseTurnLeft"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()
        state.consume_turn(self.value[0])

        return state


class ShuffleAll(ElixirLogic):
    js_alias: str = "shuffleAll"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        effect_length = len(state.effects)
        original_values = [effect.value for effect in state.effects]
        unlocked_target_indices = [
            idx for idx, effect in enumerate(state.effects) if not effect.locked
        ]
        locked_target_indices = [
            idx for idx in range(effect_length) if idx not in unlocked_target_indices
        ]

        starting = unlocked_target_indices + locked_target_indices

        shuffled_indices = RNG.shuffle(unlocked_target_indices, random_number)
        ending = shuffled_indices + locked_target_indices

        for start, end in zip(starting, ending):
            state.set_effect_count(start, original_values[end])

        return state


class SetEnchantTargetAndAmount(ElixirLogic):
    js_alias: str = "setEnchantTargetAndAmount"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()
        mutations = []
        for target in targets:
            mutations.extend(
                [
                    Mutation(
                        target=MutationTarget.prob,
                        index=target,
                        value=10000,
                        remain_turn=self.remain_turn,
                    ),
                    Mutation(
                        target=MutationTarget.enchant_increase_amount,
                        index=-1,
                        value=self.value[0],
                        remain_turn=self.remain_turn,
                    ),
                ]
            )

        for mutation in mutations:
            state.add_mutation(mutation)

        return state


class UnlockAndLockOther(ElixirLogic):
    js_alias: str = "unlockAndLockOther"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()

        effect_length = len(state.effects)

        unlocked_target_indices = [
            idx for idx, effect in enumerate(state.effects) if not effect.locked
        ]
        locked_target_indices = [
            idx for idx in range(effect_length) if idx not in unlocked_target_indices
        ]

        will_unlock = RNG.pick(locked_target_indices, random_number)
        will_lock = RNG.pick(
            unlocked_target_indices, RNG.chained_sample(random_number + 0.5)
        )  # 0.5 can be any float; it has no meaning

        state.lock(will_lock)
        state.unlock(will_unlock)

        return state


## TODO: implement this
class ChangeEffect(ElixirLogic):
    js_alias: str = "changeEffect"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        return state.deepcopy()


class LockTarget(ElixirLogic):
    js_alias: str = "lockTarget"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        if len(targets) != 1:
            raise TargetSizeMismatchException

        state = state.deepcopy()
        target = targets[0]

        state.lock(target)

        return state
