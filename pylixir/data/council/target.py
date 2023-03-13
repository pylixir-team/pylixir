from typing import Optional

from pylixir.core.base import RNG, GameState
from pylixir.core.council import CouncilTargetType, TargetSelector


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

        return RNG(random_number).shuffle(mutable_indices)[: self.count]


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
        return RNG(random_number).shuffle(candidates)[: self.count]


class MaxValueSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.maxValue

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        if self.target_condition != 0:
            raise InvalidSelectionException("Invalid proposed selector")

        availabla_indices = state.effect_board.mutable_indices()
        minimum_value = max(
            [state.effect_board.get(idx).value for idx in availabla_indices]
        )

        candidates = [
            idx
            for idx in availabla_indices
            if state.effect_board.get(idx).value == minimum_value
        ]  # since tatget_condition starts with 1
        return RNG(random_number).shuffle(candidates)[: self.count]


class UserSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.userSelect

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        if effect_index is None:
            raise InvalidSelectionException("User Selector requires effect_index")

        return [effect_index]


class LteValueSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.lteValue

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        availabla_indices = state.effect_board.mutable_indices()

        return [
            idx
            for idx in availabla_indices
            if state.effect_board.get(idx).value <= self.target_condition
        ]


class OneThreeFiveSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.oneThreeFive

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        return [0, 2, 4]


class TwoFourSelector(TargetSelector):
    type: CouncilTargetType = CouncilTargetType.twoFour

    def select_targets(
        self, state: GameState, effect_index: Optional[int], random_number: float
    ) -> list[int]:
        return [1, 3]
