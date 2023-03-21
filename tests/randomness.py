from random import Random
from typing import Any, Union

from pylixir.core.base import Randomness


class DeterministicRandomness(Randomness):
    def __init__(self, random_number: Union[int, float, list[float]]) -> None:
        if isinstance(random_number, (float, int)):
            self._random_number_set = [random_number]
        else:
            self._random_number_set = random_number

        self._idx = 0

    def binomial(self, prob: float) -> bool:
        return self._random_number < prob

    def uniform_int(self, min_range: int, max_range: int) -> int:
        bin_size = max_range - min_range + 1
        return int(self._random_number * bin_size) + min_range

    def shuffle(self, values: list[int]) -> list[int]:
        result = list(values)
        Random(self._random_number).shuffle(result)
        self._mix()
        return result

    def pick(self, values: list[int]) -> int:
        shuffled = self.shuffle(values)
        return shuffled[0]

    def _mix(self) -> None:
        self._random_number_set = [v + 1 for v in self._random_number_set]

    def weighted_sampling(self, probs: list[float]) -> int:
        if sum(probs) == 0:
            raise ValueError("Summation of probability cannot be 0")

        pivot = self._random_number * sum(probs)
        cum_prob = 0.0

        for idx, prob in enumerate(probs):
            cum_prob += prob

            if cum_prob >= pivot and prob != 0:
                return idx

        raise ValueError("random_number cannot be 1")

    def weighted_sampling_target(self, probs: list[float], target: list[Any]) -> int:
        return target[self.weighted_sampling(probs)]

    @property
    def _random_number(self) -> float:
        value = self._random_number_set[self._idx]
        self._idx = (self._idx + 1) % len(self._random_number_set)
        return value
