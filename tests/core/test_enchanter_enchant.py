import pytest

from pylixir.core.base import Enchanter
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize(
    "enchant_random_numbers, expected",
    [
        ((0.19, 0.35), [1, 0, 0, 0, 0]),
        ((0.31, 0.35), [0, 1, 0, 0, 0]),
        ((0.40, 0.35), [0, 1, 0, 0, 0]),
        ((0.41, 0.35), [0, 0, 1, 0, 0]),
        ((0.89, 0.35), [0, 0, 0, 0, 1]),
    ],
)
def test_single_enchant(
    clean_enchanter: Enchanter,
    enchant_random_numbers: list[float],
    expected: list[int],
) -> None:
    result = clean_enchanter.get_enchant_result(
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        1,
        1,
        DeterministicRandomness(enchant_random_numbers),
    )
    assert result == expected


@pytest.mark.parametrize(
    "enchant_random_numbers, expected",
    [
        ((0.19, 0.35), [1, 0, 0, 0, 0]),
        ((0.31, 0.35), [1, 0, 0, 0, 0]),
        ((0.40, 0.35), [0, 1, 0, 0, 0]),
        ((0.41, 0.35), [0, 1, 0, 0, 0]),
        ((0.89, 0.35), [0, 0, 0, 0, 1]),
    ],
)
def test_complex_prob_enchant(
    clean_enchanter: Enchanter,
    enchant_random_numbers: list[float],
    expected: list[int],
) -> None:
    result = clean_enchanter.get_enchant_result(
        [0.34, 0.13, 0.1, 0.1, 0.33],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        1,
        1,
        DeterministicRandomness(enchant_random_numbers),
    )
    assert result == expected


@pytest.mark.parametrize(
    "enchant_random_numbers, expected",
    [
        ((0.19, 0.03), [2, 0, 0, 0, 0]),
        ((0.31, 0.03), [2, 0, 0, 0, 0]),
        ((0.40, 0.03), [0, 2, 0, 0, 0]),
        ((0.41, 0.03), [0, 2, 0, 0, 0]),
        ((0.89, 0.03), [0, 0, 0, 0, 2]),
    ],
)
def test_lucky_enchant(
    clean_enchanter: Enchanter,
    enchant_random_numbers: list[float],
    expected: list[int],
) -> None:
    result = clean_enchanter.get_enchant_result(
        [0.34, 0.13, 0.1, 0.1, 0.33],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        1,
        1,
        DeterministicRandomness(enchant_random_numbers),
    )
    assert result == expected


@pytest.mark.parametrize(
    "enchant_random_numbers, expected",
    [
        ((0.19, 0.35, 0.19, 0.35), [1, 1, 0, 0, 0]),
        ((0.19, 0.35, 0.89, 0.35), [1, 0, 0, 0, 1]),
        ((0.19, 0.05, 0.19, 0.35), [2, 1, 0, 0, 0]),
        ((0.19, 0.05, 0.19, 0.05), [2, 2, 0, 0, 0]),
    ],
)
def test_multiple_enchant(
    clean_enchanter: Enchanter,
    enchant_random_numbers: list[float],
    expected: list[int],
) -> None:
    result = clean_enchanter.get_enchant_result(
        [0.34, 0.13, 0.1, 0.1, 0.33],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        2,
        1,
        DeterministicRandomness(enchant_random_numbers),
    )
    assert result == expected


@pytest.mark.parametrize(
    "enchant_random_numbers, expected",
    [
        ((0.19, 0.35), [2, 0, 0, 0, 0]),
        ((0.31, 0.35), [0, 2, 0, 0, 0]),
        ((0.40, 0.35), [0, 2, 0, 0, 0]),
        ((0.41, 0.35), [0, 0, 2, 0, 0]),
        ((0.89, 0.35), [0, 0, 0, 0, 2]),
    ],
)
def test_more_enchant(
    clean_enchanter: Enchanter,
    enchant_random_numbers: list[float],
    expected: list[int],
) -> None:
    result = clean_enchanter.get_enchant_result(
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        1,
        2,
        DeterministicRandomness(enchant_random_numbers),
    )
    assert result == expected


@pytest.mark.parametrize(
    "enchant_random_numbers, prob, expected",
    [
        ((0.19, 0.35), [0.25, 0.25, 0.25, 0.25, 0], [1, 0, 0, 0, 0]),
        ((0.89, 0.35), [0.25, 0.25, 0.25, 0.25, 0], [0, 0, 0, 1, 0]),
        ((0.35, 0.35), [0.25, 0, 0.25, 0.25, 0], [0, 0, 1, 0, 0]),
        ((0.35, 0.35), [0, 0.25, 0.25, 0.25, 0], [0, 0, 1, 0, 0]),
        ((0.35, 0.35), [0, 0, 0.33, 0.33, 0.34], [0, 0, 0, 1, 0]),
        ((0.35, 0.35), [0, 0.33, 0.33, 0.34, 0], [0, 0, 1, 0, 0]),
        ((0.19, 0.35), [1.0, 0, 0, 0, 0], [1, 0, 0, 0, 0]),
        ((0.19, 0.35), [0, 0, 1.0, 0, 0], [0, 0, 1, 0, 0]),
        ((0.19, 0.35), [0, 0, 0, 0, 1.0], [0, 0, 0, 0, 1]),
    ],
)
def test_zero(
    clean_enchanter: Enchanter,
    enchant_random_numbers: list[float],
    prob: list[float],
    expected: list[int],
) -> None:
    result = clean_enchanter.get_enchant_result(
        prob,
        [0.1, 0.1, 0.1, 0.1, 0.1],
        1,
        1,
        DeterministicRandomness(enchant_random_numbers),
    )
    assert result == expected
