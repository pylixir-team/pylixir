import pytest

from pylixir.application.council import CouncilType
from pylixir.core.committee import Sage
from pylixir.core.randomness import SeededRandomness
from pylixir.core.state import GameState
from pylixir.data.council_pool import ConcreteCouncilPool
from pylixir.data.pool import get_ingame_council_pool


@pytest.fixture(name="council_pool")
def fixture_council_pool() -> ConcreteCouncilPool:
    return get_ingame_council_pool(skip=True)


def test_pool_size_exact(council_pool: ConcreteCouncilPool) -> None:
    assert len(council_pool) == 294


@pytest.mark.parametrize(
    "council_type, count",
    [
        (CouncilType.chaos, 0),
        (CouncilType.chaosLock, 0),
        (CouncilType.lawful, 0),
        (CouncilType.lawfulLock, 0),
        (CouncilType.common, 0),
        (CouncilType.lock, 0),
        (CouncilType.exhausted, 0),
    ],
)
def test_get_council(
    council_pool: ConcreteCouncilPool, council_type: CouncilType, count: int
) -> None:
    councils = council_pool.get_available_councils(1, council_type)

    assert len(councils) > count


def test_sample_council(
    council_pool: ConcreteCouncilPool, abundant_state: GameState
) -> None:
    for seed in range(50):
        randomness = SeededRandomness(seed)
        council_pool.sample_council(
            abundant_state, Sage(power=2, is_removed=False, slot=1), randomness, []
        )
