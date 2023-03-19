from pylixir.core.base import Board, GameState, Randomness


def choose_max_indices(
    board: Board, randomness: Randomness, count: int = 1
) -> list[int]:
    availabla_indices = board.mutable_indices()
    minimum_value = max([board.get(idx).value for idx in availabla_indices])

    candidates = [
        idx for idx in availabla_indices if board.get(idx).value == minimum_value
    ]  # since tatget_condition starts with 1
    return randomness.shuffle(candidates)[:count]


def choose_min_indices(
    board: Board, randomness: Randomness, count: int = 1
) -> list[int]:
    availabla_indices = board.mutable_indices()
    minimum_value = min([board.get(idx).value for idx in availabla_indices])

    candidates = [
        idx for idx in availabla_indices if board.get(idx).value == minimum_value
    ]  # since tatget_condition starts with 1
    return randomness.shuffle(candidates)[:count]
