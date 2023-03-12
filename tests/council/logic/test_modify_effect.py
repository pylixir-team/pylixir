import pytest

from pylixir.core.base import GameState
from pylixir.council.logic import IncreaseTargetWithRatio, TargetSizeMismatchException
from tests.council.logic.util import assert_effect_changed


@pytest.mark.parametrize(
    "ratio, random_number, success",
    [
        (1500, 3500, False),
        (3500, 2000, True),
        (10000, 9999.999999, True),
        (2500, 2500.1, False),
    ],
)
def test_increase_target_with_ratio(
    random_number: float, ratio: int, success: bool, abundant_state: GameState
) -> None:
    target_index = 0
    amount = 1
    logic = IncreaseTargetWithRatio(
        ratio=ratio,
        value=(amount, 0),
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, [target_index], random_number)
    assert_effect_changed(
        abundant_state, changed_state, target_index, amount if success else 0
    )


def test_increase_target_with_ratio_reject_multiple_target(abundant_state):
    with pytest.raises(TargetSizeMismatchException):
        IncreaseTargetWithRatio(
            ratio=2500,
            value=(1, 0),
            remain_turn=1,
        ).reduce(abundant_state, [2, 3], 2344)
