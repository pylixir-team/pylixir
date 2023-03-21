import pydantic

from pylixir.application.council import Council
from pylixir.core.base import Board, Effect, Enchanter
from pylixir.core.committee import Sage, SageCommittee
from pylixir.core.progress import Progress
from pylixir.core.state import GameState


class CommitteeView(pydantic.BaseModel):
    committee: SageCommittee
    councils: list[Council]

    def show(
        self,
    ) -> str:
        return "\n".join(
            self._show_as_text(self.committee.sages[idx], self.councils[idx])
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

        return "[______]"


class BoardView(pydantic.BaseModel):
    board: Board
    enchanter: Enchanter

    def show(self) -> str:
        enchant_probs = self.enchanter.query_enchant_prob(self.board.locked_indices())
        lucky_ratios = self.enchanter.query_lucky_ratio()

        return "\n".join(
            f"{idx}: {self._get_effect_repr(self.board.get(idx))}   {enchant_probs[idx]*100:.2f}% | [{lucky_ratios[idx]*100:.0f}%]"
            for idx in range(5)
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

        return f"[{effect.value}] {joined}"


class ProgressView(pydantic.BaseModel):
    progress: Progress

    def show(self) -> str:
        return f"Turn left: {self.progress.get_turn_left()} | reroll left: {self.progress.get_reroll_left()}"


class ClientView(pydantic.BaseModel):
    state: GameState
    councils: list[Council]

    def represent_as_text(
        self,
    ) -> str:
        council_view = CommitteeView(
            committee=self.state.committee,
            councils=self.councils,
        ).show()
        board_view = BoardView(
            board=self.state.board, enchanter=self.state.enchanter
        ).show()
        progress_view = ProgressView(progress=self.state.progress).show()

        return f"""
{board_view}
{progress_view}

{council_view}
        """