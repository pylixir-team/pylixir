from typing import Callable

from pylixir.application.reducer import (
    PickCouncilAndEnchantAndRerollAction,
    pick_council,
)
from pylixir.application.view import ClientView
from pylixir.core.base import Randomness
from pylixir.core.state import GameState
from pylixir.data.council_pool import ConcreteCouncilPool


class Client:
    def __init__(
        self,
        state_initializer: Callable[[], GameState],
        state: GameState,
        council_pool: ConcreteCouncilPool,
        randomness: Randomness,
    ):
        self._state_initializer = state_initializer
        self._state = state
        self._council_pool = council_pool
        self._randomness = randomness
        self._state.suggestions = self._council_pool.get_council_queries(
            state, randomness, is_reroll=False
        )

    def run(self, action: PickCouncilAndEnchantAndRerollAction) -> None:
        self._state = self._run(self._state, action)

    def _run(
        self, state: GameState, action: PickCouncilAndEnchantAndRerollAction
    ) -> GameState:
        return pick_council(
            action,
            state,
            self._randomness,
            self._council_pool,
        )

    def get_view(self) -> ClientView:
        return ClientView(
            state=self._state,
            councils=[
                self._council_pool.get_council(query)
                for query in self._state.suggestions
            ],
        )

    def get_state(self) -> GameState:
        return self._state.deepcopy()