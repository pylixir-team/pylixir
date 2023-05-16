from typing import Optional

from pylixir.application.council import Council
from pylixir.application.terminal.board import show_board
from pylixir.application.terminal.councils import show_councils
from pylixir.application.terminal.progress import show_progress
from pylixir.core.base import Board
from pylixir.core.state import GameState


def show_game_state(
    state: GameState, councils: list[Council], previous_board: Optional[Board] = None
) -> str:
    return f"""
{show_board(state.board, state.enchanter, previous_board)}
{show_progress(state.progress)}

{show_councils(state.committee, councils) if state.progress.turn_left > 0 else "DONE"}
    """
