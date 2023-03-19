from pylixir.application.state import GameState


def assert_effect_changed(
    source: GameState,
    target: GameState,
    effect_index: int,
    amount: int,
) -> None:
    if amount == 0:
        assert source == target
    else:
        source.board.modify_effect_count(effect_index=effect_index, amount=amount)
        assert source == target
