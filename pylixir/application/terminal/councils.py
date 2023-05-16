from pylixir.application.council import Council
from pylixir.core.committee import Sage, SageCommittee


def _get_sage_repr(sage: Sage) -> str:
    if sage.power < 0:
        chaos_string = "X" * abs(sage.power)
        return f"[{chaos_string:_<6}]"
    if sage.power > 0:
        chaos_string = "O" * abs(sage.power)
        return f"[{chaos_string:_<3}]   "

    return "[______]"


def _show_sage_with_council(sage: Sage, council: Council) -> str:
    sage_repr = _get_sage_repr(sage)
    council_repr = council.descriptions[sage.slot]
    return f"""{sage_repr} | {council_repr}"""


def show_councils(committee: SageCommittee, councils: list[Council]) -> str:
    return "\n".join(
        _show_sage_with_council(committee.sages[idx], councils[idx]) for idx in range(3)
    )
