import abc
import random
import time

from pylixir.application.query import GameStateView
from pylixir.application.reducer import PickCouncilAndEnchantAndRerollAction
from pylixir.interface.cli import ClientBuilder


class ElixirModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feed(self, view: GameStateView) -> PickCouncilAndEnchantAndRerollAction:
        ...


class RandomModel(metaclass=abc.ABCMeta):
    def feed(self, view: GameStateView) -> PickCouncilAndEnchantAndRerollAction:
        sage_index = random.choice(view.get_valid_sage_indices())
        effect_index = random.choice(view.get_valid_effect_indices())
        return PickCouncilAndEnchantAndRerollAction(
            effect_index=effect_index, sage_index=sage_index
        )


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def measure(self, view: GameStateView) -> float:
        ...


class Any53Metric(Metric):
    def measure(self, view: GameStateView) -> float:
        effects = view.board.get_effect_values()
        a, b = view.board.unlocked_indices()

        return float(effects[a] + effects[b] >= 18)


class Evaluator:
    def __init__(self, model: ElixirModel, metric: Metric, verbose=False) -> None:
        self._model = model
        self._metric = metric
        self._verbose = verbose
        self._client_builder = ClientBuilder()

    def evaluate(self, seed: int) -> float:
        client = self._client_builder.get_client(seed)

        while True:

            action = self._model.feed(client.get_view())
            if self._verbose:
                print("-------")
                print(client.text())
                print(action)

            client.run(action)
            if client.get_state().progress.get_turn_left() == 0:  # may replaced by view
                break

        return self._metric.measure(client.get_view())

    def benchmark(self, count: int = 10000, interval: int = 100):
        random.seed(42)
        start = time.time()

        metric_sum = 0
        count_sum = 0
        for seed in range(count):
            count_sum += 1
            metric_sum += self.evaluate(seed)

            if (count_sum) % interval == 0:
                time_elapsed = time.time() - start
                print(
                    f"{count_sum} | Average metric: {metric_sum / count_sum} | Elapsed {time_elapsed:.2f}s"
                )
