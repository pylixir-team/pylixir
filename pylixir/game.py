import pydantic

from pylixir.core.base import RNG, Decision, GameState
from pylixir.core.council import Council, CouncilRepository


class Client(pydantic.BaseModel):
    def __init__(
        self,
        council_repository: CouncilRepository,
        rng: RNG,
    ):
        self._council_repository = council_repository
        self._rng = rng

    def run(self, state: GameState, decision: Decision) -> GameState:
        # council_id = state.council_ids[decision.sage_index]
        # council = self._council_repository.get_council(council_id)

        targets = council.target_selector.select_targets(
            state, decision.effect_index, self.step_rng()
        )
        new_state = council.logic.reduce(state, targets, self.step_rng())

        new_state.enchant(self._rng.sample())

        # new_state.replace_council_ids(self._council_repository.sample(new_state))

        return new_state

    def step_rng(self) -> float:
        forked_rng = self._rng.fork()
        return forked_rng.sample()
