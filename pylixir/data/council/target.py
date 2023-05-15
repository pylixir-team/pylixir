import enum
from typing import Optional, Type

from pylixir.application.council import TargetSelector
from pylixir.core.base import Randomness
from pylixir.core.state import GameState
from pylixir.data.council.common import choose_max_indices, choose_min_indices


class InvalidSelectionException(Exception):
    ...


class NoneSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        return []

    def is_valid(self, state: GameState) -> bool:
        return True


class RandomSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        mutable_indices = state.board.mutable_indices()

        return randomness.shuffle(mutable_indices)[: self.count]

    def is_valid(self, state: GameState) -> bool:
        return True


class ProposedSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        if self.target_condition == 0:
            raise InvalidSelectionException("Invalid proposed selector")

        return [self.target_index]  # since tatget_condition starts with 1

    def is_valid(self, state: GameState) -> bool:
        return self.target_index in state.board.mutable_indices()

    @property
    def target_index(self) -> int:
        return self.target_condition - 1


class MinValueSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        if self.target_condition != 0:
            raise InvalidSelectionException("Invalid proposed selector")

        return choose_min_indices(state.board, randomness, count=self.count)

    def is_valid(self, state: GameState) -> bool:
        return True


class MaxValueSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        if self.target_condition != 0:
            raise InvalidSelectionException("Invalid proposed selector")

        return choose_max_indices(state.board, randomness, count=self.count)

    def is_valid(self, state: GameState) -> bool:
        return True


class UserSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        if effect_index is None:
            raise InvalidSelectionException("User Selector requires effect_index")

        return [effect_index]

    def is_valid(self, state: GameState) -> bool:
        return True


class LteValueSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        availabla_indices = state.board.mutable_indices()

        return [
            idx
            for idx in availabla_indices
            if state.board.get(idx).value <= self.target_condition
        ]

    def is_valid(self, state: GameState) -> bool:
        return True


class OneThreeFiveSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        return [0, 2, 4]

    def is_valid(self, state: GameState) -> bool:
        return True


class TwoFourSelector(TargetSelector):
    def select_targets(
        self, state: GameState, effect_index: Optional[int], randomness: Randomness
    ) -> list[int]:
        return [1, 3]

    def is_valid(self, state: GameState) -> bool:
        return True


class CouncilTargetType(enum.Enum):
    none = "none"
    random = "random"
    proposed = "proposed"
    maxValue = "maxValue"
    minValue = "minValue"
    userSelect = "userSelect"
    lteValue = "lteValue"
    oneThreeFive = "oneThreeFive"
    twoFour = "twoFour"


def get_target_classes() -> dict[str, Type[TargetSelector]]:
    return {
        CouncilTargetType.none.value: NoneSelector,
        CouncilTargetType.random.value: RandomSelector,
        CouncilTargetType.proposed.value: ProposedSelector,
        CouncilTargetType.minValue.value: MinValueSelector,
        CouncilTargetType.maxValue.value: MaxValueSelector,
        CouncilTargetType.userSelect.value: UserSelector,
        CouncilTargetType.lteValue.value: LteValueSelector,
        CouncilTargetType.oneThreeFive.value: OneThreeFiveSelector,
        CouncilTargetType.twoFour.value: TwoFourSelector,
    }
