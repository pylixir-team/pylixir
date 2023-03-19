from pylixir.core.council import CouncilPool
from pylixir.interface.game import Client
from pylixir.core.randomness import SeededRandomness
from pylixir.data.pool import get_ingame_council_pool
from pylixir.interface.configuration import state_initializer, create_empty_committee
from pylixir.core.base import Decision, GameState, Randomness


def cli():
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


    while True:
        try:
            sage_index, effect_index = map(int, input().strip().split())
        except ValueError:
            print("Wrong input. Try again as {int, int}")
            continue

        if sage_index > 2 or effect_index > 4:
            print("Sage index < 3 and Effect index < 5")
            continue

        print(f"Decide to Sage {sage_index} with effect {effect_index}")
    
        client.run(Decision(sage_index=sage_index, effect_index=effect_index))
        print(client.get_view().represent_as_text())


if __name__ == "__main__":
    cli()
