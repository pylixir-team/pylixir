import pydantic

from pylixir.core.base import RNG, Decision, GameState
from pylixir.council.base import Council, CouncilRepository


class Client(pydantic.BaseModel):
    def __init__(
        self,
        council_repository: CouncilRepository,
        rng: RNG,
    ):
        self._council_repository = council_repository
        self._rng = rng

    def run(self, state: GameState, decision: Decision) -> GameState:
        council = state.councils[decision.sage_index]
        mutation = council.pick(decision.effect_index, self._rng.sample())

        state.add_mutation(mutation)
        state.enchant(self._rng.sample())

        state.councils = self._council_repository.sample(state)

        return state
