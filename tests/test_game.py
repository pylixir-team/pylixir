from pylixir.core.base import Decision, GameState, Randomness
from pylixir.core.council import CouncilPool
from pylixir.core.randomness import SeededRandomness
from pylixir.data.pool import get_ingame_council_pool
from pylixir.interface.configuration import create_empty_committee, state_initializer
from pylixir.interface.game import Client
from pylixir.interface.view import ClientView


def test_game():
    seed = 42
    council_pool = get_ingame_council_pool()
    initial_state = state_initializer()
    randomness = SeededRandomness(seed)
    committee = create_empty_committee(
        council_pool,
        initial_state,
        randomness,
    )
    client = Client(
        state_initializer,
        initial_state,
        council_pool=council_pool,
        committee=committee,
        randomness=randomness,
    )

    print(client.get_view().represent_as_text())

    client.run(Decision(sage_index=1, effect_index=1))
