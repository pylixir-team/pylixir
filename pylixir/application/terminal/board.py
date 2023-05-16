from typing import Optional

from pylixir.application.terminal.color import bcolors
from pylixir.core.base import Board, Effect, Enchanter


def _get_effect_repr(effect: Effect, prvious_effect: Optional[Effect]) -> str:
    template = list("__1__2_345")
    number_idx = [idx for idx in range(len(template)) if template[idx] in "12345"]

    for idx in range(effect.value):
        if idx == effect.value - 1 and idx in number_idx:
            continue

        template[idx] = "X"

    joined = ",".join(template)
    if effect.value > 0:
        joined = joined[: effect.value * 2 - 1] + "]" + joined[effect.value * 2 :]

    colored = ""

    if effect.locked:
        colored = f"{bcolors.GRAY}{joined}{bcolors.ENDC}"
    elif prvious_effect:
        for idx, char in enumerate(joined):
            offset = (idx + 2) // 2
            if prvious_effect.value < (offset) <= effect.value:
                colored += f"{bcolors.GREEN}{char}{bcolors.ENDC}"
            elif effect.value < (offset) <= prvious_effect.value:
                colored += f"{bcolors.RED}{char}{bcolors.ENDC}"
            elif offset <= effect.value:
                colored += f"{bcolors.YELLOW}{char}{bcolors.ENDC}"
            else:
                colored += f"{char}"

    return f"[{effect.value}] {colored}"


def show_board(
    board: Board, enchanter: Enchanter, previous_board: Optional[Board]
) -> str:
    enchant_probs = enchanter.query_enchant_prob(board.locked_indices())
    lucky_ratios = enchanter.query_lucky_ratio()

    return "\n".join(
        f"{idx}: {_get_effect_repr(board.get(idx), previous_board.get(idx) if previous_board else None)}\
  {enchant_probs[idx]*100:.2f}% | [{lucky_ratios[idx]*100:.0f}%]"
        for idx in range(5)
    )
