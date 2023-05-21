from pylixir.interface.cli import get_client


def test_pick() -> None:
    client = get_client(42)
    print(client.view())

    client.pick(sage_index=1, effect_index=1)

    print(client.view())


def test_reroll() -> None:
    client = get_client(42)

    client.reroll()
