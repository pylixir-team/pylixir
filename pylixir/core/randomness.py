from random import Random

from pylixir.core.base import Randomness


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
        )
