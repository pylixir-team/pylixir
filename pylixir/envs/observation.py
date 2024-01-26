import enum
from typing import Union

import pydantic

from pylixir.application.game import Client
from pylixir.application.reducer import PickCouncilAndEnchantAndRerollAction
from pylixir.core.base import Board, Enchanter
from pylixir.core.committee import Sage, SageCommittee
from pylixir.core.progress import Progress
from pylixir.core.state import CouncilQuery
from pylixir.envs.feature import get_feature_builder


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
        (EmbeddingName.suggestion, 20),
        (EmbeddingName.committee, 3),
        (EmbeddingName.progress, 2),
        (EmbeddingName.board, 5),
        (EmbeddingName.enchanter, 10),
    ]

    def get_order_offset(self, name: EmbeddingName) -> tuple[int, int]:
        order_map = {}
        offset = 0
        for embedding_name, size in self.order:
            order_map[embedding_name] = (offset, size)
            offset += size

        return order_map[name]


class EmbeddingProvider:
    """This wil create such integer-set, which may suitable and parsed by  EmbeddingRenderer"""

    def __init__(self, index_map: dict[str, int]) -> None:
        self._council_id_map = index_map
        self._feature_builder = get_feature_builder()
        self._index_to_action: list[PickCouncilAndEnchantAndRerollAction] = sum(
            [
                [
                    PickCouncilAndEnchantAndRerollAction(
                        effect_index=effect_index, sage_index=sage_index
                    )
                    for sage_index in range(3)
                ]
                for effect_index in range(5)
            ],
            [],
        )

        self._suggestion_embedding_keys = list(
            sorted(self._feature_builder.get_feature_by_id("31000").keys())
        )

    def action_index_to_action(
        self, action_index: int
    ) -> PickCouncilAndEnchantAndRerollAction:
        return self._index_to_action[action_index]

    def create_observation(self, client: Client) -> list[int]:
        state = client.get_state()

        committee_vector = self._committee_to_vector(state.committee)  # 3

        progress_vector = self._progress_to_vector(state.progress)  # 2
        board_vector = self._board_to_vector(state.board)  # 5

        enchanter_vector = self._enchanter_to_vector(
            state.enchanter, state.board.locked_indices()
        )  # 10, x1000
        suggestion_vector = self._suggestions_to_vector(state.suggestions)  # 3 x []

        return (
            committee_vector
            + progress_vector
            + board_vector
            + enchanter_vector
            + suggestion_vector
        )

    def current_total_reward(self, client: Client) -> float:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()

        first, second = values[0], values[1]
        if 0 not in alived_indices:
            first = 0
        if 1 not in alived_indices:
            second = 0

        return float(2**first + 2**second)

    def current_valuation(
        self, client: Client, index: tuple[int, int] = (0, 1)
    ) -> float:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()

        i, j = index
        first, second = values[i], values[j]

        if i not in alived_indices:
            first = 0
        if j not in alived_indices:
            second = 0

        return first + second

    def is_complete(self, client: Client, threshold: int) -> bool:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()
        valid_values = [values[idx] for idx in alived_indices]
        valid_values = sorted(valid_values)
        second_largest, largest = valid_values[-2:]
        return (largest + second_largest) >= threshold

    def _board_to_vector(self, board: Board) -> list[int]:
        effect_count = board.get_effect_values()

        return effect_count

    def _enchanter_to_vector(
        self, enchanter: Enchanter, locked: list[int]
    ) -> list[int]:
        lucky_ratio = enchanter.query_lucky_ratio()
        enchant_prob = enchanter.query_enchant_prob(locked)

        return [int(v * 100) for v in lucky_ratio + enchant_prob]

    def _progress_to_vector(self, progress: Progress) -> list[int]:
        return [progress.turn_left, progress.reroll_left]

    def _sage_to_integer(self, sage: Sage) -> int:
        if sage.is_removed:
            return 0
        return sage.power + 7  # 1 ~ 10

    def _committee_to_vector(self, committee: SageCommittee) -> list[int]:
        sage_indices = [self._sage_to_integer(sage) for sage in committee.sages]
        return sage_indices

    def _suggestions_to_vector(
        self, suggestions: tuple[CouncilQuery, CouncilQuery, CouncilQuery]
    ) -> list[int]:
        council_vector = []
        for council in suggestions:
            feature = self._feature_builder.get_feature_by_id(council.id)
            council_vector += [feature[k] for k in self._suggestion_embedding_keys]

        return council_vector


class DictObservation:
    def __init__(self, index_map: dict[str, int]) -> None:
        self._council_id_map = index_map
        self._feature_builder = get_feature_builder()
        self._index_to_action: list[PickCouncilAndEnchantAndRerollAction] = sum(
            [
                [
                    PickCouncilAndEnchantAndRerollAction(
                        effect_index=effect_index, sage_index=sage_index
                    )
                    for sage_index in range(3)
                ]
                for effect_index in range(5)
            ],
            [],
        )

        self._suggestion_embedding_keys = list(
            sorted(self._feature_builder.get_feature_by_id("31000").keys())
        )

    def action_index_to_action(
        self, action_index: int
    ) -> PickCouncilAndEnchantAndRerollAction:
        return self._index_to_action[action_index]

    def create_observation(self, client: Client) -> dict[str, Union[int, list[float]]]:
        state = client.get_state()

        vector: dict[str, Union[int, list[float]]] = {}
        vector.update(self._committee_to_vector(state.committee))  # 3

        vector.update(self._progress_to_vector(state.progress))  # 2
        vector.update(self._board_to_vector(state.board))  # 5

        vector.update(
            self._enchanter_to_vector(state.enchanter, state.board.locked_indices())
        )  # 10, x1000
        vector.update(self._suggestions_to_vector(state.suggestions))  # 3 x []

        return vector

    def current_total_reward(self, client: Client) -> float:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()

        first, second = values[0], values[1]
        if 0 not in alived_indices:
            first = 0
        if 1 not in alived_indices:
            second = 0

        return first + second

    def current_valuation(
        self, client: Client, index: tuple[int, int] = (0, 1)
    ) -> float:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()

        i, j = index
        first, second = values[i], values[j]

        if i not in alived_indices:
            first = 0
        if j not in alived_indices:
            second = 0

        return first + second

    def is_complete(self, client: Client, threshold: int) -> bool:
        state = client.get_state()
        values = state.board.get_effect_values()
        alived_indices = state.board.unlocked_indices()
        valid_values = [values[idx] for idx in alived_indices]
        valid_values = sorted(valid_values)
        second_largest, largest = valid_values[-2:]
        return (largest + second_largest) >= threshold

    def _board_to_vector(self, board: Board) -> dict[str, int]:
        effect_count = board.get_effect_values()
        locked_indices = board.locked_indices()

        lock_added_effect_count = [
            (11 if idx in locked_indices else cnt)
            for idx, cnt in enumerate(effect_count)
        ]

        return {f"board_{idx}": cnt for idx, cnt in enumerate(lock_added_effect_count)}

    def _enchanter_to_vector(
        self, enchanter: Enchanter, locked: list[int]
    ) -> dict[str, list[float]]:
        return {
            "enchant_lucky": enchanter.query_lucky_ratio(),
            "enchant_prob": enchanter.query_enchant_prob(locked),
        }

    def _progress_to_vector(self, progress: Progress) -> dict[str, int]:
        return {"turn_left": progress.turn_left, "reroll": progress.reroll_left}

    def _sage_to_integer(self, sage: Sage) -> int:
        if sage.is_removed:
            return 0
        return sage.power + 7  # 1 ~ 10

    def _committee_to_vector(self, committee: SageCommittee) -> dict[str, int]:
        return {
            f"committee_{idx}": self._sage_to_integer(sage)
            for idx, sage in enumerate(committee.sages)
        }

    def _suggestions_to_vector(
        self, suggestions: tuple[CouncilQuery, CouncilQuery, CouncilQuery]
    ) -> dict[str, int]:
        council_vector = {}
        for idx, council in enumerate(suggestions):
            feature = self._feature_builder.get_feature_by_id(council.id)
            council_vector.update(
                {
                    f"suggestion_{idx}_{k}": feature[k]
                    for k in self._suggestion_embedding_keys
                }
            )

        return council_vector
