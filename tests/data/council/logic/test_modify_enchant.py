from pylixir.core.base import GameState
from pylixir.data.council.operation import (
    SetEnchantEffectCount,
    SetEnchantIncreaseAmount,
)
from tests.randomness import DeterministicRandomness


def test_set_enchant_increase_amount(abundant_state: GameState) -> None:
    enchant_amount = 2
    operation = SetEnchantIncreaseAmount(
        ratio=0, value=(enchant_amount, 0), remain_turn=1
    )

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )

    assert enchant_amount == changed_state.enchanter.get_enchant_amount()


def test_set_effect_enchant_count(abundant_state: GameState) -> None:
    enchant_count = 2
    operation = SetEnchantEffectCount(ratio=0, value=(enchant_count, 0), remain_turn=1)

    changed_state = operation.reduce(
        abundant_state, [], DeterministicRandomness(0.3456)
    )

    assert enchant_count == changed_state.enchanter.get_enchant_effect_count()
