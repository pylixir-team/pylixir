from typing import Optional

from pylixir.core.base import RNG, GameState
from pylixir.council.base import TargetSelector


class NoneSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        return []


class RandomSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        mutable_indices = state.effect_board.mutable_indices()

        return RNG.shuffle(mutable_indices, random_number)[: self.count]
