from pylixir.core.base import Enchanter
from tests.core.util import is_equal_in_fp_precision


def test_clean_prob(clean_enchanter: Enchanter) -> None:
    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([]), [0.2, 0.2, 0.2, 0.2, 0.2]
    )


def test_increase_once(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(0, 0.1, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([]), [0.3, 0.175, 0.175, 0.175, 0.175]
    )


def test_increase_and_decrease_once(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(0, 0.1, 1)
    clean_enchanter.mutate_prob(0, -0.1, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([]), [0.2, 0.2, 0.2, 0.2, 0.2]
    )


def test_full_increase(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(0, 1, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([]), [1.0, 0, 0, 0, 0]
    )


def test_full_decrease(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(0, -1, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([]), [0, 0.25, 0.25, 0.25, 0.25]
    )


def test_complex_1(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(3, 0.05, 1)
    clean_enchanter.mutate_prob(1, -0.05, 1)
    clean_enchanter.mutate_prob(0, -0.1, 1)
    clean_enchanter.mutate_prob(3, 0.7, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([]),
        [0.00020925, 0.00032679, 0.00047304, 0.99851787, 0.00047304],
    )


def test_complex_2(clean_enchanter: Enchanter) -> None:
    clean_enchanter.mutate_prob(2, -0.05, 1)

    assert is_equal_in_fp_precision(
        clean_enchanter.query_enchant_prob([0, 3, 4]), [0, 0.55, 0.45, 0, 0]
    )
