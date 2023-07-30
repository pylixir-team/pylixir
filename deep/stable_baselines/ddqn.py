import torch

from deep.stable_baselines.train import train
from deep.stable_baselines.util import ModelSettings, get_basic_train_settings
from deep.stable_baselines.policy.council_feature import CustomCombinedExtractor
from deep.stable_baselines.policy.ddqn import DDQN

class DQNModelSettings(ModelSettings):
    ...


train_envs = get_basic_train_settings(name="DDQN")
train_envs.update(
    {
        "expname": "exp-neg-decay-b128-emb-majorkey",
        "total_timesteps": int(15e5),
        "checkpoint_freq": int(10e4),
        "eval_freq": int(10e4),
        "n_envs": 4
    }
)

model_envs: DQNModelSettings = {
    "policy": "MultiInputPolicy",
    "learning_rate": 0.0003,
    "seed": 37,
    "kwargs": {
        "batch_size": 128,
        "tau": 0.5,
        "gamma": 0.99,
        "train_freq": 4,
        "tensorboard_log": "./logs/tb/",
        "verbose": 1,
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [128, 128],
            "features_extractor_class":CustomCombinedExtractor,
            "features_extractor_kwargs":dict(),
        },
        "tensorboard_log": "./logs/tb/",
        "verbose": 1,
    },
}


if __name__ == "__main__":
    train(train_envs, model_envs, DDQN)
