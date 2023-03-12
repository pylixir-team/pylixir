from typing import Optional

from pylixir.core.base import RNG, GameState
from pylixir.council.base import CouncilTargetType, TargetSelector


class InvalidSelectionException(Exception):
    ...


class NoneSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.none

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        return []


class RandomSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.random

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        mutable_indices = state.effect_board.mutable_indices()

        return RNG.shuffle(mutable_indices, random_number)[: self.count]


class ProposedSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.proposed

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        if self.target_condition == 0:
            raise InvalidSelectionException("Invalid proposed selector")

        return [self.target_condition - 1]  # since tatget_condition starts with 1


class MinValueSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.minValue

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        if self.target_condition != 0:
            raise InvalidSelectionException("Invalid proposed selector")

        availabla_indices = state.effect_board.mutable_indices()
        minimum_value = min(
            [state.effect_board.get(idx).value for idx in availabla_indices]
        )

        candidates = [
            idx
            for idx in availabla_indices
            if state.effect_board.get(idx).value == minimum_value
        ]  # since tatget_condition starts with 1
        return RNG.shuffle(candidates, random_number)[: self.count]
