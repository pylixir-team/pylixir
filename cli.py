from pylixir.data.council_pool import ConcreteCouncilPool
from pylixir.application.game import Client
from pylixir.core.randomness import SeededRandomness
from pylixir.data.pool import get_ingame_council_pool
from pylixir.interface.configuration import state_initializer, create_empty_committee
from pylixir.application.reducer import PickCouncilAndEnchantAndRerollAction
from pylixir.interface.cli import get_client

def tui(client: Client):
    while True:
        try:
            sage_index, effect_index = map(int, input().strip().split())
        except ValueError:
            print("Wrong input. Try again as {int, int}")
            continue

        if sage_index > 2 or effect_index > 4:
            print("Sage index < 3 and Effect index < 5")
            continue


        view = client.get_view().councils[sage_index].descriptions[sage_index]
        print(f"Decide to Sage {sage_index} with effect {effect_index} | {view}")
    
        client.run(PickCouncilAndEnchantAndRerollAction(sage_index=sage_index, effect_index=effect_index))
        print(client.get_view().represent_as_text())


if __name__ == "__main__":
    tui(get_client(42))
