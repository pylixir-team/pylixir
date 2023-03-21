import abc

from pylixir.application.council import Council
from pylixir.core.base import Randomness
from pylixir.core.state import CouncilQuery, GameState


class CouncilPool(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_council_queries(
        self, state: GameState, randomness: Randomness, is_reroll: bool = False
    ) -> tuple[CouncilQuery, CouncilQuery, CouncilQuery]:
        ...

    @abc.abstractmethod
    def get_council(self, query: CouncilQuery) -> Council:
        ...
