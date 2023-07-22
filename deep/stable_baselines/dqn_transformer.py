from typing import Any
import torch
from stable_baselines3 import DQN

from deep.stable_baselines.train import train
from deep.stable_baselines.util import ModelSettings, get_basic_train_settings
from deep.stable_baselines.policy.council_feature import CustomCombinedExtractor
from deep.stable_baselines.policy.transformer_network import TransformerQPolicy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

class DQNModelSettings(ModelSettings):
    ...


train_envs = get_basic_train_settings(name="DQN")
train_envs.update(
    {
        "expname": "transformer-L1-H8-Emb128-lrdecay3e-4",
        "total_timesteps": int(20e5),
        "checkpoint_freq": int(10e4),
        "eval_freq": int(10e4),
        "n_envs": 4
    }
)


class LearningRateDecay():
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"LearningRateDecay:<{self.start}>-><{self.end}>"        

    def __call__(self, progress: float) -> float:
        rate = self.end / self.start
        progress = 1 - progress

        if progress < 0.2:
            return self.start * (5 * progress)

        progress = (progress - 0.2) * 1.25

        return self.start * (rate ** progress)


model_envs: DQNModelSettings = {
    "policy": TransformerQPolicy,
    "learning_rate": LearningRateDecay(3e-4, 3e-5),
    "seed": 37,
    "kwargs": {
        "batch_size": 128,
        "tau": 0.5,
        "gamma": 0.99,
        "train_freq": 4,
        "tensorboard_log": "./logs/tb/",
        "verbose": 1,
        "policy_kwargs": {
            "transformer_layers": 6,
            "vector_size": 128,
            "hidden_dimension": 128,
            "transformer_heads": 8,
            "features_extractor_class":CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "prob_hidden_dim": 16,
                "suggesion_feature_hidden_dim": 16,
                "embedding_dim": 128,
                "flatten_output": False
            },
        },
        "tensorboard_log": "./logs/tb/",
        "verbose": 1,
    },
}


if __name__ == "__main__":
    train(train_envs, model_envs, DQN) #, continue_from="./logs/checkpoints/DQN.transformer-L2-H4-Emb128-lrdecay3e-4/rl_model_1000000_steps.zip"
