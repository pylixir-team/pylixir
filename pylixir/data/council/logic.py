from pylixir.core.base import RNG, GameState, Mutation, MutationTarget
from pylixir.core.council import ElixirLogic


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
            state.effect_board.modify_effect_count(target, self.value[0])

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

        state.effect_board.modify_effect_count(target, diff)

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

        original_values = state.effect_board.get_effect_values()
        unlocked_indices = state.effect_board.unlocked_indices()
        locked_indices = state.effect_board.locked_indices()

        starting = unlocked_indices + locked_indices

        shuffled_indices = RNG(random_number).shuffle(unlocked_indices)
        ending = shuffled_indices + locked_indices

        for start, end in zip(starting, ending):
            state.effect_board.set_effect_count(start, original_values[end])

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

        rng = RNG(random_number)

        will_unlock = rng.fork().pick(state.effect_board.locked_indices())
        will_lock = rng.fork().pick(state.effect_board.unlocked_indices())

        state.effect_board.lock(will_lock)
        state.effect_board.unlock(will_unlock)

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

        state.effect_board.lock(target)

        return state


class IncreaseReroll(ElixirLogic):
    js_alias: str = "increaseReroll"

    def reduce(
        self, state: GameState, targets: list[int], random_number: float
    ) -> GameState:
        state = state.deepcopy()
        state.modify_reroll(self.value[0])

        return state
