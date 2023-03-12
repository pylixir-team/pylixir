from pylixir.core.base import GameState, Mutation, MutationTarget
from pylixir.council.base import ElixirLogic


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

    def reduce(self, state: GameState, targets: list[int], random_number: float):
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
