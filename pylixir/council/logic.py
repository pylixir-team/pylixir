from pylixir.core.base import GameState, Mutation, MutationTarget
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
