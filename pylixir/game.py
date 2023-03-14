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

        targets = council.target_selector.select_targets(
            state, decision.effect_index, self.step_rng()
        )
        new_state = council.logic.reduce(state, targets, self.step_rng())

        enchanted_result = new_state.enchanter.enchant(
            state.board.locked_indices(), self._rng.sample()
        )
        for idx, amount in enumerate(enchanted_result):
            new_state.board.modify_effect_count(idx, amount)

        # council_id = state.council_ids[decision.sage_index]
        # council = self._council_repository.get_council(council_id)

        committee.pick(decision.sage_index)

        return new_state

    def step_rng(self) -> float:
        forked_rng = self._rng.fork()
        return forked_rng.sample()
