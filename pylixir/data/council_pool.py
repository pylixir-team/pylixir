from typing import cast

from pylixir.application.council import Council, CouncilType
from pylixir.application.service import CouncilPool
from pylixir.core.base import Randomness
from pylixir.core.committee import Sage
from pylixir.core.state import CouncilQuery, GameState

CouncilSet = tuple[Council, Council, Council]


class ConcreteCouncilPool(CouncilPool):
    def __init__(
        self, councils: list[Council], trials_before_exact_sampling: int = 5
    ) -> None:
        self._councils = councils
        self._council_id_map = {council.id: council for council in self._councils}
        self._trials_before_exact_sampling = trials_before_exact_sampling
        self._council_type_cache = {}
        for sage_slot in range(3):
            for council_type in [
                CouncilType.lawfulLock,
                CouncilType.lawful,
                CouncilType.chaosLock,
                CouncilType.chaos,
                CouncilType.common,
                CouncilType.lock,
                CouncilType.exhausted,
            ]:
                self._council_type_cache[
                    (sage_slot, council_type)
                ] = self._compute_available_councils(sage_slot, council_type)

    def __len__(self) -> int:
        return len(self._councils)

    def get_index_map(self) -> dict[str, int]:
        return {self._councils[idx].id: idx for idx in range(len(self))}

    def get_council_queries(
        self, state: GameState, randomness: Randomness, is_reroll: bool = False
    ) -> tuple[CouncilQuery, CouncilQuery, CouncilQuery]:
        council_set = self.get_council_set(
            state, state.committee.sages, randomness, is_reroll=is_reroll
        )

        queries = cast(
            tuple[CouncilQuery, CouncilQuery, CouncilQuery],
            tuple(CouncilQuery(id=c.id) for c in council_set),
        )
        return queries

    def get_council(self, query: CouncilQuery) -> Council:
        return self._council_id_map[query.id].copy()

    def get_council_set(
        self,
        state: GameState,
        sages: tuple[Sage, Sage, Sage],
        randomness: Randomness,
        is_reroll: bool = False,
    ) -> CouncilSet:
        # TODO: protection logic for reroll
        sage_a, sage_b, sage_c = sages

        if is_reroll:
            council_1 = self.sample_council(
                state, sage_a, randomness, [state.suggestions[0].id]
            )
            council_2 = self.sample_council(
                state, sage_b, randomness, [state.suggestions[1].id, council_1.id]
            )
            council_3 = self.sample_council(
                state,
                sage_c,
                randomness,
                [state.suggestions[2].id, council_1.id, council_2.id],
            )
        else:
            council_1 = self.sample_council(state, sage_a, randomness, [])
            council_2 = self.sample_council(state, sage_b, randomness, [council_1.id])
            council_3 = self.sample_council(
                state, sage_c, randomness, [council_1.id, council_2.id]
            )

        return (council_1, council_2, council_3)

    def sample_council(
        self,
        state: GameState,
        sage: Sage,
        randomness: Randomness,
        forbidden_council_ids: list[str],
    ) -> Council:
        def _is_valid(target_council: Council) -> bool:
            return (
                target_council.is_valid(state)
                and target_council.id not in forbidden_council_ids
            )

        council_type = self._get_council_type(state, sage)
        candidates, weights = self.get_available_councils(sage.slot, council_type)

        for _ in range(self._trials_before_exact_sampling):
            council = randomness.weighted_sampling_target(weights, candidates)
            if _is_valid(council):
                return council

        refined_council = [council for council in candidates if _is_valid(council)]

        refined_weights = [float(council.pickup_ratio) for council in refined_council]
        return randomness.weighted_sampling_target(refined_weights, refined_council)

    def get_available_councils(
        self, sage_slot: int, council_type: CouncilType
    ) -> tuple[list[Council], list[float]]:
        cache = self._council_type_cache.get((sage_slot, council_type))
        if cache:
            return cache

        return self._compute_available_councils(sage_slot, council_type)

    def _compute_available_councils(
        self, sage_slot: int, council_type: CouncilType
    ) -> tuple[list[Council], list[float]]:
        councils = [
            council
            for council in self._councils
            if council.type == council_type and (council.slot_type in (3, sage_slot))
        ]
        weights = [float(council.pickup_ratio) for council in councils]
        return councils, weights

    def _get_council_type(self, state: GameState, sage: Sage) -> CouncilType:
        if sage.is_removed:
            return CouncilType.exhausted

        if sage.is_lawful_max():
            if state.requires_lock():
                return CouncilType.lawfulLock

            return CouncilType.lawful

        if sage.is_chaos_max():
            if state.requires_lock():
                return CouncilType.chaosLock

            return CouncilType.chaos

        if state.requires_lock():
            return CouncilType.lock

        return CouncilType.common
