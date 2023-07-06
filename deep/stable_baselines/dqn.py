from stable_baselines3 import DQN
import torch

from deep.stable_baselines.train import train
from deep.stable_baselines.util import get_basic_train_settings, ModelSettings


class DQNModelSettings(ModelSettings):
    ...


train_envs = get_basic_train_settings(name="DQN")

model_envs: DQNModelSettings = {
    "policy": "MlpPolicy", 
    "learning_rate": 0.0001,
    "kwargs": {
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 4,
        'policy_kwargs': {
            'activation_fn': torch.nn.ReLU,
            'net_arch': [64, 64]
        }
    }
}


if __name__ == "__main__":
    train(train_envs, model_envs, DQN)
