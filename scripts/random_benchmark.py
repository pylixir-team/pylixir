from pylixir.ml.base import Evaluator, RandomModel, Any53Metric

from cProfile import Profile
from pstats import Stats

def test():
    evaluator = Evaluator(
        RandomModel(),
        Any53Metric()
    )

    evaluator.benchmark(4000, interval=1000)

profiler = Profile()
profiler.run('test()')

stats = Stats(profiler)
stats.strip_dirs()
stats.sort_stats('cumtime')
stats.print_stats()


#if __name__ == "__main__":
