from pylixir.data.repository import get_ingame_council_repository


def test_loading() -> None:
    repository = get_ingame_council_repository(skip=True)
