import pytest

from pylixir.application.state import GameState
from pylixir.core.base import Mutation, MutationTarget
from pylixir.data.council.operation import (
    MutateLuckyRatio,
    MutateProb,
    SetEnchantTargetAndAmount,
)
from tests.randomness import DeterministicRandomness


@pytest.mark.parametrize("random_number", [0, 0.3, 1.0])
@pytest.mark.parametrize("indices", [[1], [2, 4], [3]])
def test_mutate_prob(
    random_number: float, indices: list[int], abundant_state: GameState
) -> None:
    operation = MutateProb(
        ratio=0,
        value=(3500, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, indices, DeterministicRandomness(random_number)
    )
    expected_mutations = [
        Mutation(target=MutationTarget.prob, index=index, value=3500, remain_turn=1)
        for index in indices
    ]

    for mutation in expected_mutations:
        abundant_state.enchanter.apply_mutation(mutation)

    assert abundant_state == changed_state


@pytest.mark.parametrize("random_number", [0, 0.3, 1.0])
@pytest.mark.parametrize("indices", [[1], [2, 4], [3]])
def test_mutate_lucky_ratio(
    random_number: float, indices: list[int], abundant_state: GameState
) -> None:
    operation = MutateLuckyRatio(
        ratio=0,
        value=(3500, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, indices, DeterministicRandomness(random_number)
    )
    expected_mutations = [
        Mutation(
            target=MutationTarget.lucky_ratio, index=index, value=3500, remain_turn=1
        )
        for index in indices
    ]
    for mutation in expected_mutations:
        abundant_state.enchanter.apply_mutation(mutation)

    assert abundant_state == changed_state


@pytest.mark.parametrize("random_number", [0, 0.3, 1.0])
@pytest.mark.parametrize("indices", [[1], [2], [3]])
def test_enchant_target_and_amount(
    random_number: float, indices: list[int], abundant_state: GameState
) -> None:
    amount = 2
    operation = SetEnchantTargetAndAmount(
        ratio=0,
        value=(amount, 0),
        remain_turn=1,
    )

    changed_state = operation.reduce(
        abundant_state, indices, DeterministicRandomness(random_number)
    )
    expected_mutations: list[Mutation] = sum(
        [
            [
                Mutation(
                    target=MutationTarget.prob, index=index, value=1.0, remain_turn=1
                ),
                Mutation(
                    target=MutationTarget.enchant_increase_amount,
                    index=-1,
                    value=amount,
                    remain_turn=1,
                ),
            ]
            for index in indices
        ],
        [],
    )

    for mutation in expected_mutations:
        abundant_state.enchanter.apply_mutation(mutation)

    assert abundant_state == changed_state
