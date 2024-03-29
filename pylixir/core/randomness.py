from random import Random
from typing import TypeVar

from pylixir.core.base import Randomness

T = TypeVar("T")


class SeededRandomness(Randomness):
    def __init__(self, seed: float) -> None:
        self._rng = Random(seed)

    def binomial(self, prob: float) -> bool:
        return self._rng.random() < prob

    def uniform_int(self, min_range: int, max_range: int) -> int:
        return self._rng.randint(min_range, max_range)

    def shuffle(self, values: list[int]) -> list[int]:
        results = list(values)
        self._rng.shuffle(results)
        return results

    def pick(self, values: list[int]) -> int:
        return self.shuffle(values)[0]

    def weighted_sampling(self, probs: list[float]) -> int:
        return self._rng.choices(
            list(range(len(probs))),
            weights=probs,
            k=1,
        )[0]

    def weighted_sampling_target(self, probs: list[float], target: list[T]) -> T:
        return self._rng.choices(
            target,
            weights=probs,
            k=1,
        )[0]
