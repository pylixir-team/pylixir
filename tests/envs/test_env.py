import gymnasium as gym

from pylixir.envs import register_env


def test_pylixir_env() -> None:
    register_env()
    env = gym.make("pylixir/PylixirEnv-v0")
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(4)
    assert observation.shape == (512,)
    # env.close()
