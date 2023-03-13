from pylixir.core.base import GameState, Mutation


def assert_mutation_extended(
    source: GameState, target: GameState, mutations: list[Mutation]
) -> None:
    ref = list(source.mutations)
    ref.extend(mutations)
    assert ref == target.mutations


def assert_effect_changed(
    source: GameState,
    target: GameState,
    effect_index: int,
    amount: int,
) -> None:
    if amount == 0:
        assert source == target
    else:
        source.effect_board.modify_effect_count(
            effect_index=effect_index, amount=amount
        )
        assert source == target
