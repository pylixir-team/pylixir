from typing import Callable, Optional

from pylixir.application.council import Council, ForbiddenActionException
from pylixir.application.reducer import (
    PickCouncilAndEnchantAndRerollAction,
    pick_council,
)
from pylixir.application.terminal.view import show_game_state
from pylixir.core.base import Board, Randomness
from pylixir.core.state import GameState
from pylixir.data.council_pool import ConcreteCouncilPool


class Client:
    def __init__(
        self,
        state_initializer: Callable[[], GameState],
        state: GameState,
        council_pool: ConcreteCouncilPool,
        randomness: Randomness,
        show_previous_board: bool = False,
    ):
        self._state_initializer = state_initializer
        self._state = state
        self._council_pool = council_pool
        self._randomness = randomness
        self._state.suggestions = self._council_pool.get_council_queries(
            state, randomness, is_reroll=False
        )
        self._show_previous_board = show_previous_board
        self._previous_board: Optional[Board] = None

    def run(self, action: PickCouncilAndEnchantAndRerollAction) -> bool:
        if self._show_previous_board:
            self._previous_board = self.get_state().board.copy(deep=True)

        try:
            self._state = self._run(self._state, action)
        except ForbiddenActionException:
            return False

        return True

    def _run(
        self, state: GameState, action: PickCouncilAndEnchantAndRerollAction
    ) -> GameState:
        return pick_council(
            action,
            state,
            self._randomness,
            self._council_pool,
        )

    def get_current_councils(self) -> list[Council]:
        return [
            self._council_pool.get_council(query) for query in self._state.suggestions
        ]

    def view(self) -> str:
        return show_game_state(
            self._state, self.get_current_councils(), self._previous_board
        )

    def get_state(self) -> GameState:
        return self._state

    def is_done(self) -> bool:
        return self._state.progress.get_turn_left() == 0
