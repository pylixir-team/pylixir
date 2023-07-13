from gymnasium.envs.registration import register

from pylixir.envs.DictPylixirEnv import DictPylixirEnv
from pylixir.envs.PylixirEnv import PylixirEnv


def register_env() -> None:
    register(
        id="pylixir/PylixirEnv-v0",
        entry_point="pylixir.envs:PylixirEnv",
        max_episode_steps=300,
    )

    register(
        id="pylixir/DictPylixirEnv-v0",
        entry_point="pylixir.envs:DictPylixirEnv",
        max_episode_steps=300,
    )
