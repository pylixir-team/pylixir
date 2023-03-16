from pylixir.core.base import Decision, GameState, Randomness
from pylixir.core.council import SageCommittee


class Client:
    def __init__(
        self,
        state: GameState,
        committee: SageCommittee,
        randomness: Randomness,
    ):
        self._state = state
        self._committee = committee
        self._randomness = randomness

    def run(
        self, state: GameState, committee: SageCommittee, decision: Decision
    ) -> GameState:
        council = committee.get_council(decision.sage_index)

        for logic in council.logics:
            state = logic.apply(state, decision, self._randomness)

        enchanted_result = state.enchanter.enchant(
            state.board.locked_indices(), self._randomness
        )
        for idx, amount in enumerate(enchanted_result):
            state.board.modify_effect_count(idx, amount)

        # council_id = state.council_ids[decision.sage_index]
        # council = self._council_repository.get_council(council_id)

        committee.pick(decision.sage_index)

        return state
