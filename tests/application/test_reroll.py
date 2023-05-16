import pytest

from pylixir.application.reducer import reroll
from pylixir.core.state import GameState
from pylixir.data.council_pool import ConcreteCouncilPool
from pylixir.data.pool import get_ingame_council_pool
from tests.randomness import DeterministicRandomness


@pytest.fixture(name="council_pool")
def fixture_council_pool() -> ConcreteCouncilPool:
    return get_ingame_council_pool(skip=True)


def test_reroll(step_state: GameState, council_pool: ConcreteCouncilPool) -> None:
    updated_state = reroll(step_state, DeterministicRandomness(0.1), council_pool)

    first_suggestion = [query.id for query in updated_state.suggestions]

    updated_state.progress.modify_reroll(1)
    updated_state = reroll(updated_state, DeterministicRandomness(0.1), council_pool)

    second_suggestion = [query.id for query in updated_state.suggestions]

    for first, second in zip(first_suggestion, second_suggestion):
        assert first != second


def test_reroll_when_option_restricted(
    step_state: GameState, council_pool: ConcreteCouncilPool
) -> None:
    step_state.board.lock(0)
    step_state.board.lock(1)

    updated_state = reroll(
        step_state,
        DeterministicRandomness([idx * 1e-3 for idx in range(200)]),
        council_pool,
    )

    prev_suggestion = [query.id for query in updated_state.suggestions]

    for _ in range(10):
        updated_state.progress.modify_reroll(1)
        updated_state = reroll(
            updated_state, DeterministicRandomness(0.1), council_pool
        )

        current_suggestion = [query.id for query in updated_state.suggestions]

        for prev, curr in zip(prev_suggestion, current_suggestion):
            assert prev != curr

        prev_suggestion = current_suggestion
