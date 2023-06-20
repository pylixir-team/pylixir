
from pylixir.interface.cli import ClientBuilder
from pylixir.application.game import Client
from pylixir.core.state import GameState
from pylixir.core.base import Board
from pylixir.core.committee import SageCommittee
from pylixir.core.progress import Progress
from pylixir.application.council import Council
from pylixir.application.service import CouncilPool
import numpy as np
from pylixir.core.base import Board, Enchanter, Randomness
from pylixir.core.committee import SageCommittee, Sage
from pylixir.core.progress import Progress
from pylixir.application.reducer import (
    PickCouncilAndEnchantAndRerollAction,
    pick_council,
)
import pydantic

import enum


class EmbeddingName(enum.Enum):
    suggestion = "suggestion"
    committee = "committee"

    progress = "progress"
    board = "board"
    enchanter = "enchanter"


class EmbeddingConfiguration(pydantic.BaseModel):
    turn_embedding_size: int
    reroll_embedding_size: int
    sage_embedding_size: int
    council_embedding_size: int

    action_dim: int = 15

    order: list[tuple[EmbeddingName, int]] = [
        (EmbeddingName.suggestion, 3),
        (EmbeddingName.committee, 3),
        (EmbeddingName.progress, 2),
        (EmbeddingName.board, 5),
        (EmbeddingName.enchanter, 10),
    ]

    def get_order_offset(self, name: EmbeddingName):
        order_map = {}
        offset = 0
        for embedding_name, size in self.order:
            order_map[embedding_name] = (offset, size)
            offset += size

        return order_map[name]


class EmbeddingProvider:
    """This wil create such integer-set, which may suitable and parsed by  EmbeddingRenderer
    """
    def __init__(self, index_map: dict[str, int]) -> None:
        self._council_id_map = index_map

        self._index_to_action: list[PickCouncilAndEnchantAndRerollAction] = sum([
            [PickCouncilAndEnchantAndRerollAction(effect_index=effect_index, sage_index=sage_index) for sage_index in range(3)]
            for effect_index in range(5)
        ], [])

    def action_index_to_action(self, action_index: int)-> PickCouncilAndEnchantAndRerollAction:
        return self._index_to_action[action_index]

    def board_to_vector(self, board: Board):
        effect_count = board.get_effect_values()
        
        return effect_count

    def enchanter_to_vector(self, enchanter: Enchanter, locked: list[int]):
        lucky_ratio = enchanter.query_lucky_ratio()
        enchant_prob = enchanter.query_enchant_prob(locked)

        return [int(v * 100) for v in lucky_ratio + enchant_prob]

    def progress_to_vector(self, progress: Progress):
        return [progress.turn_left, progress.reroll_left]

    def sage_to_integer(self, sage: Sage):
        if sage.is_removed:
            return 0
        return sage.power + 7 # 1 ~ 10 

    def committee_to_vector(self, committee: SageCommittee):
        sage_indices = [self.sage_to_integer(sage) for sage in committee.sages]
        return sage_indices

    def suggestions_to_vector(self, suggestions: tuple[Council, Council, Council]) -> list[int]:
        council_vector = [
            self._council_id_map[council.id] for council in suggestions
        ]
        return council_vector

    def create_observation(self, client: Client) -> list[int]:
        state = client.get_state()

        suggestion_vector = self.suggestions_to_vector(state.suggestions) # 3
        committee_vector = self.committee_to_vector(state.committee) # 3

        progress_vector = self.progress_to_vector(state.progress) # 2
        board_vector = self.board_to_vector(state.board) # 5

        enchanter_vector = self.enchanter_to_vector(state.enchanter, state.board.locked_indices()) # 10, x1000

        return (suggestion_vector + committee_vector + progress_vector + board_vector + enchanter_vector)

    def current_total_reward(self, client: Client) -> float:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()
        #alived_indices = [idx for idx in state.board.unlocked_indices() if idx in (0, 1)]
        first, second = values[0], values[1]
        if 0 not in alived_indices:
            first = 0
        if 1 not in alived_indices:
            second = 0
        # valid_values = [values[idx] for idx in alived_indices]
        # first, second = sorted(valid_values)[-2:]
        return first + second
        # return sum((x/5) ** 2 for x in valid_values)

    def is_complete(self, client: Client, threshold: int) -> bool:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()
        valid_values = [values[idx] for idx in alived_indices]
        valid_values = sorted(valid_values)
        second_largest, largest = valid_values[-2:]
        return (largest + second_largest) >= threshold
