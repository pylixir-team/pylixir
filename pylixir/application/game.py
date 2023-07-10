from typing import Callable, Dict, Optional

from pylixir.application.council import Council, ForbiddenActionException
from pylixir.application.reducer import (
    PickCouncilAndEnchantAndRerollAction,
    pick_council,
    reroll,
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

    def pick(self, sage_index: int, effect_index: int) -> bool:
        """
        Pick council from `sage_index` with selecting effect at `effect_index`.
        `effect_index` will be ignored when selected Logic is not `proposed`.

        returns False if choosen action is forbidden. This may occur when selecting
        locked effect or exhausted sage.
        """
        if sage_index not in self._state.committee.get_valid_slots():
            return False

        self._set_previous_board_as_now()

        try:
            self._state = pick_council(
                PickCouncilAndEnchantAndRerollAction(
                    effect_index=effect_index,
                    sage_index=sage_index,
                ),
                self._state,
                self._randomness,
                self._council_pool,
            )
        except ForbiddenActionException:
            return False

        return True

    def reroll(self) -> bool:
        """
        Reroll current suggestion and spent one reroll chance.
        return False if reroll action is forbidden. This may occur hwne there is no reroll action left.
        """
        self._set_previous_board_as_now()
        if self._state.progress.reroll_left <= 0:
            return False
        self._state = reroll(
            self._state,
            self._randomness,
            self._council_pool,
        )
        return True

    def get_current_councils(self) -> list[Council]:
        """
        Get current Councils with full metadata.
        You may use this method to access full information about current suggestion.
        """
        return [
            self._council_pool.get_council(query) for query in self._state.suggestions
        ]

    def view(self) -> str:
        """
        Get terminal-friendly output to display current state.
        """
        return show_game_state(
            self._state, self.get_current_councils(), self._previous_board
        )

    def get_state(self) -> GameState:
        """
        Get current state. This returns reference of current state; Changing returned value may affect
        to game state.
        """
        return self._state

    def is_done(self) -> bool:
        """
        Returns whether game is done.
        """
        return self._state.progress.get_turn_left() == 0

    def _set_previous_board_as_now(self) -> None:
        if self._show_previous_board:
            self._previous_board = self.get_state().board.copy(deep=True)

    def get_council_pool_index_map(self) -> Dict[str, int]:
        """
        Get council index map.
        """
        return self._council_pool.get_index_map()
