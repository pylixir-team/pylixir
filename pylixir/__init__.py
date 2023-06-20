from gymnasium.envs.registration import register

register(
    id="pylixir/PylixirEnv-v0",
    entry_point="pylixir.envs:PylixirEnv",
    max_episode_steps=300,
)
