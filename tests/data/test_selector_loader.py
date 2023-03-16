from typing import Optional

import pytest

from pylixir.core.base import GameState, Randomness
from pylixir.core.council import TargetSelector
from pylixir.data.loader import ElixirTargetSelectorLoader


class DummySelectorA(TargetSelector):
    def is_valid(self, state: GameState) -> bool:
        return True

    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        return []


class DummySelectorB(TargetSelector):
    def is_valid(self, state: GameState) -> bool:
        return True

    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        return []


def test_loading_from_operation_loader() -> None:
    selector_loader = ElixirTargetSelectorLoader(
        {
            "A": DummySelectorA,
            "B": DummySelectorB,
        }
    )

    selector_loader.get_selector("A", 0, 0)
    selector_loader.get_selector("B", 0, 0)

    with pytest.raises(KeyError):
        selector_loader.get_selector("dummySelectorA", 0, 0)
