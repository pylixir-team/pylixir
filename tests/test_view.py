from pylixir.application.reducer import PickCouncilAndEnchantAndRerollAction
from pylixir.interface.cli import get_client


def test_game() -> None:
    client = get_client(42)
    print(client.text().represent_as_text())

    client.run(PickCouncilAndEnchantAndRerollAction(sage_index=1, effect_index=1))

    print(client.text().represent_as_text())
