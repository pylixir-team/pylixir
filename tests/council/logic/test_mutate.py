import pytest

from pylixir.core.base import GameState, Mutation, MutationTarget
from pylixir.council.logic import MutateLuckyRatio, MutateProb
from tests.council.logic.util import assert_mutation_extended


@pytest.mark.parametrize("random_number", [0, 0.3, 1.0])
@pytest.mark.parametrize("indices", [[1], [2, 4], [3]])
def test_mutate_prob(
    random_number: float, indices: list[int], abundant_state: GameState
) -> None:
    logic = MutateProb(
        ratio=0,
        value=(3500, 0),
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, indices, random_number)
    expected_mutations = [
        Mutation(target=MutationTarget.prob, index=index, value=3500, remain_turn=1)
        for index in indices
    ]

    assert_mutation_extended(abundant_state, changed_state, expected_mutations)


@pytest.mark.parametrize("random_number", [0, 0.3, 1.0])
@pytest.mark.parametrize("indices", [[1], [2, 4], [3]])
def test_mutate_lucky_ratio(
    random_number: float, indices: list[int], abundant_state: GameState
) -> None:
    logic = MutateLuckyRatio(
        ratio=0,
        value=(3500, 0),
        remain_turn=1,
    )

    changed_state = logic.reduce(abundant_state, indices, random_number)
    expected_mutations = [
        Mutation(
            target=MutationTarget.lucky_ratio, index=index, value=3500, remain_turn=1
        )
        for index in indices
    ]

    assert_mutation_extended(abundant_state, changed_state, expected_mutations)
