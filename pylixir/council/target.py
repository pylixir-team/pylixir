from typing import Optional

from pylixir.core.base import Decision, GameState
from pylixir.council.base import TargetSelector


class NoneSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int]
    ) -> list[int]:
        return []


class RandomSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int]
    ) -> list[int]:
        return []
