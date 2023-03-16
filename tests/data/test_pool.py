from pylixir.data.pool import get_ingame_council_pool


def test_loading() -> None:
    pool = get_ingame_council_pool(skip=True)
