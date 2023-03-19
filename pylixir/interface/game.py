from typing import Callable

from pylixir.application.council import CouncilPool, SageCommittee
from pylixir.application.state import GameState
from pylixir.core.base import Decision, Randomness
from pylixir.interface.view import ClientView


class Client:
    def __init__(
        self,
        state_initializer: Callable[[], GameState],
        state: GameState,
        council_pool: CouncilPool,
        committee: SageCommittee,
        randomness: Randomness,
    ):
        self._state_initializer = state_initializer
        self._state = state
        self._council_pool = council_pool
        self._committee = committee
        self._randomness = randomness

    def run(self, decision: Decision):
        self._state = self._run(self._state, decision)

    def _run(self, state: GameState, decision: Decision) -> GameState:
        council = self._committee.get_council(decision.sage_index)

        for logic in council.logics:
            state = logic.apply(state, decision, self._randomness)

        enchanted_result = state.enchanter.enchant(
            state.board.locked_indices(), self._randomness
        )

        for idx, amount in enumerate(enchanted_result):
            state.board.modify_effect_count(idx, amount)

        self._committee.pick(decision.sage_index)

        new_council_set = self._council_pool.get_council_set(
            state,
            sages=self._committee.sages,
            randomness=self._randomness,
            is_reroll=False,
        )
        self._committee.set_councils(new_council_set)
        # council_id = state.council_ids[decision.sage_index]
        # council = self._council_repository.get_council(council_id)
        return state

    def get_view(self) -> ClientView:
        return ClientView(
            state=self._state,
            committee=self._committee,
        )
