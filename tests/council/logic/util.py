from pylixir.core.base import (
    GameState,
    Mutation,
)


def assert_mutation_extended(
    source: GameState, target: GameState, mutations: list[Mutation]
) -> None:
    ref = list(source.mutations)
    ref.extend(mutations)
    assert ref == target.mutations
