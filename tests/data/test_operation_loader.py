import pytest

from pylixir.application.council import ElixirOperation
from pylixir.application.state import GameState
from pylixir.core.base import Randomness
from pylixir.data.loader import ElixirOperationLoader


class DummyOperationA(ElixirOperation):
    def is_valid(self, state: GameState) -> bool:
        return True

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state.deepcopy()


class DummyOperationB(ElixirOperation):
    def is_valid(self, state: GameState) -> bool:
        return True

    def reduce(
        self, state: GameState, targets: list[int], randomness: Randomness
    ) -> GameState:
        return state.deepcopy()


def test_loading_from_operation_loader() -> None:
    operation_loader = ElixirOperationLoader(
        [
            DummyOperationA,
            DummyOperationB,
        ]
    )

    operation_loader.get_operation("dummyOperationA", 1300, (3000, 0), 1)
    operation_loader.get_operation("dummyOperationB", 1300, (3000, 0), 1)

    with pytest.raises(KeyError):
        operation_loader.get_operation("dummyOperationNot", 1300, (3000, 0), 1)
