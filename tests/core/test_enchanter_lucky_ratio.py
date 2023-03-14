from pylixir.core.base import Enchanter
from tests.core.util import is_equal_in_fp_precision


def test_clean_lucky_ratio(clean_enchanter: Enchanter) -> None:
    assert is_equal_in_fp_precision(
        clean_enchanter.query_lucky_ratio(), [0.1, 0.1, 0.1, 0.1, 0.1]
    )


def test_increase_once(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_lucky_ratio(1, 0.07, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_lucky_ratio(), [0.1, 0.17, 0.1, 0.1, 0.1]
    )


def test_increase_and_decrease_once(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_lucky_ratio(0, 0.1, 1)
    clean_enchanter.mutate_lucky_ratio(0, 0.1, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_lucky_ratio(), [0.3, 0.1, 0.1, 0.1, 0.1]
    )


def test_full_increase(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_lucky_ratio(0, 1, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_lucky_ratio(), [1.0, 0.1, 0.1, 0.1, 0.1]
    )
