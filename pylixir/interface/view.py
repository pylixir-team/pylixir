from typing import Callable

import pydantic

from pylixir.application.council import Council, CouncilPool, Sage, SageCommittee
from pylixir.application.state import GameState
from pylixir.core.base import Board, Decision, Effect, Randomness


class CommitteeView(pydantic.BaseModel):
    committee: SageCommittee

    def show(self) -> str:
        return "\n".join(
            self._show_as_text(self.committee.sages[idx], self.committee.councils[idx])
            for idx in range(3)
        )

    def _show_as_text(self, sage: Sage, council: Council) -> str:
        sage_repr = self._get_sage_repr(sage)
        council_repr = council.descriptions[sage.slot]
        return f"""{sage_repr} | {council_repr}"""

    def _get_sage_repr(self, sage: Sage) -> str:
        if sage.power < 0:
            chaos_string = "X" * abs(sage.power)
            return f"[{chaos_string:_<6}]"
        if sage.power > 0:
            chaos_string = "O" * abs(sage.power)
            return f"[{chaos_string:_<3}]   "
        if sage.power == 0:
            return f"[______]"


class BoardView(pydantic.BaseModel):
    board: Board

    def show(self) -> str:
        return "\n".join(
            f"{idx}: {self._get_effect_repr(self.board.get(idx))}" for idx in range(5)
        )

    def _get_effect_repr(self, effect: Effect) -> str:
        template = list("__1__2_345")
        number_idx = [idx for idx in range(len(template)) if template[idx] in "12345"]
        for idx in range(effect.value):
            if idx == effect.value - 1 and idx in number_idx:
                continue

            template[idx] = "X"

        joined = ",".join(template)
        if effect.value > 0:
            joined = joined[: effect.value * 2 - 1] + "]" + joined[effect.value * 2 :]

        return joined


class ClientView(pydantic.BaseModel):
    state: GameState
    committee: SageCommittee

    def represent_as_text(self) -> str:
        council_view = CommitteeView(committee=self.committee).show()
        board_view = BoardView(board=self.state.board).show()

        return f"""
{council_view}

{board_view}
        """

        representation = f"""
        

        
        """
