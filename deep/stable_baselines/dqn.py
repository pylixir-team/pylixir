from stable_baselines3 import DQN

from deep.stable_baselines.train import train
from deep.stable_baselines.util import get_basic_train_settings, ModelSettings


class DQNModelSettings(ModelSettings):
    ...


train_envs = get_basic_train_settings(name="DQN")

model_envs: DQNModelSettings = {"policy": "MlpPolicy", "learning_rate": 0.0001}

if __name__ == "__main__":
    train(train_envs, model_envs, DQN)
