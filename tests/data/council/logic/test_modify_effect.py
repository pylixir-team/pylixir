from typing import Type

import pytest

from pylixir.core.base import GameState
from pylixir.core.council import ElixirLogic
from pylixir.data.council.logic import (
    IncreaseTargetRanged,
    IncreaseTargetWithRatio,
    TargetSizeMismatchException,
)
from tests.data.council.logic.util import assert_effect_changed


@pytest.mark.parametrize(
    "ratio, random_number, success",
    [
        (1500, 0.35, False),
        (3500, 0.20, True),
        (10000, 0.999999, True),
        (2500, 0.25001, False),
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


@pytest.mark.parametrize(
    "value_range, random_number, amount",
    [
        ((1, 2), 0.2500, 1),
        ((1, 2), 0.7500, 2),
        ((0, 2), 0.2500, 0),
        ((0, 2), 0.6000, 1),
        ((-2, 2), 0.5000, 0),
        ((-2, 2), 0.9200, 2),
        ((-2, 2), 0.2100, -1),
    ],
)
def test_increase_target_ranged(
    value_range: tuple[int, int],
    random_number: float,
    amount: int,
    abundant_state: GameState,
) -> None:
    target_index = 0
    logic = IncreaseTargetRanged(
        ratio=0,
        value=value_range,
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, [target_index], random_number)
    assert_effect_changed(
        abundant_state,
        changed_state,
        target_index,
        amount,
    )


@pytest.mark.parametrize(
    "logic_class",
    [
        IncreaseTargetWithRatio,
        IncreaseTargetRanged,
    ],
)
def test_increase_target_with_ratio_reject_multiple_target(
    logic_class: Type[ElixirLogic], abundant_state: GameState
) -> None:
    with pytest.raises(TargetSizeMismatchException):
        logic_class(
            ratio=2500,
            value=(1, 0),
            remain_turn=1,
        ).reduce(abundant_state, [2, 3], 2344)
