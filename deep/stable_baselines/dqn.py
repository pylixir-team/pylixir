import torch
from stable_baselines3 import DQN

from deep.stable_baselines.train import train
from deep.stable_baselines.util import ModelSettings, get_basic_train_settings


class DQNModelSettings(ModelSettings):
    ...


train_envs = get_basic_train_settings(name="DQN")
train_envs.update(
    {
        "run_num_pretrained": 2,
    }
)

model_envs: DQNModelSettings = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0001,
    "kwargs": {
        "batch_size": 32,
        "tau": 0.5,
        "gamma": 0.99,
        "train_freq": 4,
        "policy_kwargs": {"activation_fn": torch.nn.ReLU, "net_arch": [128, 128]},
    },
}


if __name__ == "__main__":
    train(train_envs, model_envs, DQN)
