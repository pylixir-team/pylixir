import gymnasium as gym


def test_pylixir_env() -> None:
    env = gym.make("pylixir/PylixirEnv-v0")
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(4)
    assert observation.shape == (23,)
    # env.close()
