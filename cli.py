from pylixir.application.game import Client
from pylixir.interface.cli import get_client
import fire


def tui(client: Client):
    print(client.view())
    while True:
        command = input().strip()
        
        if command == "r":
            client.reroll()
        else:
            try:
                sage_index, effect_index = map(int, command.split())
            except ValueError:
                print("Wrong input. Try again as {int, int} or r for Reroll")
                continue

            if sage_index > 2 or effect_index > 4:
                print("Sage index may < 3 and Effect index may < 5")
                continue

            client.pick(sage_index, effect_index)

        print(client.view())
        if client.is_done():
            break

def run(seed: int = 42):
    client = get_client(seed, show_previous_board=True)
    tui(client)


if __name__ == "__main__":
    fire.Fire(run)
