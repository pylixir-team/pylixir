from pylixir.application.game import Client
from pylixir.core.randomness import SeededRandomness
from pylixir.data.pool import get_ingame_council_pool
from pylixir.interface.configuration import state_initializer


def get_client(seed: int, show_previous_board: bool = False) -> Client:
    council_pool = get_ingame_council_pool()
    initial_state = state_initializer()
    randomness = SeededRandomness(seed)
    client = Client(
        state_initializer,
        initial_state,
        council_pool=council_pool,
        randomness=randomness,
        show_previous_board=show_previous_board,
    )

    return client


class ClientBuilder:
    def __init__(self) -> None:
        self._council_pool = get_ingame_council_pool()
        self._state_initializer = state_initializer

    def get_client(self, seed: float) -> Client:
        return Client(
            self._state_initializer,
            self._state_initializer(),
            council_pool=self._council_pool,
            randomness=SeededRandomness(seed),
        )
