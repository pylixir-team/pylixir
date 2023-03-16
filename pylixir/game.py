from pylixir.core.base import RNG, Decision, GameState
from pylixir.core.council import SageCommittee


class Client:
    def __init__(
        self,
        state: GameState,
        committee: SageCommittee,
        rng: RNG,
    ):
        self._state = state
        self._committee = committee
        self._rng = rng

    def run(
        self, state: GameState, committee: SageCommittee, decision: Decision
    ) -> GameState:
        council = committee.get_council(decision.sage_index)

        for logic in council.logics:
            state = logic.apply(state, decision, self._rng)

        enchanted_result = state.enchanter.enchant(
            state.board.locked_indices(), self._rng.sample()
        )
        for idx, amount in enumerate(enchanted_result):
            state.board.modify_effect_count(idx, amount)

        # council_id = state.council_ids[decision.sage_index]
        # council = self._council_repository.get_council(council_id)

        committee.pick(decision.sage_index)

        return state

    def step_rng(self) -> float:
        forked_rng = self._rng.fork()
        return forked_rng.sample()
