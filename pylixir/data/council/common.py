from pylixir.core.base import Board, Randomness


def choose_max_indices(
    board: Board, randomness: Randomness, count: int = 1
) -> list[int]:
    availabla_indices = board.mutable_indices()
    maximum_value = max([board.get(idx).value for idx in availabla_indices])

    candidates = [
        idx for idx in availabla_indices if board.get(idx).value == maximum_value
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


def choose_random_indices_with_exclusion(
    board: Board, randomness: Randomness, excludes: list[int]
) -> int:

    mutable_indices = [idx for idx in board.mutable_indices() if idx not in excludes]

    return randomness.pick(mutable_indices)
