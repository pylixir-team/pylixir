from stable_baselines3 import PPO

from deep.stable_baselines.policy.council_feature import CustomCombinedExtractor
from deep.stable_baselines.policy.ppo_transformer_network import PPOTransformerPolicy
from deep.stable_baselines.train import train
from deep.stable_baselines.util import ModelSettings, get_basic_train_settings


class PPOModelSettings(ModelSettings):
    ...


train_envs = get_basic_train_settings(name="PPO")
train_envs.update(
    {
        "expname": "transformer-L3-H4-Emb128-1e-4-disentangle-obs-mid-12",
        "total_timesteps": int(40e5),
        "checkpoint_freq": int(10e4),
        "eval_freq": int(10e4),
        "n_envs": 4,
    }
)


class LearningRateDecay:
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

        return self.start * (rate**progress)


model_envs: PPOModelSettings = {
    "policy": PPOTransformerPolicy,
    "learning_rate": 1e-4,
    "seed": 37,
    "kwargs": {
        "batch_size": 32,
        "gamma": 0.99,
        "verbose": 1,
        "policy_kwargs": {
            "transformer_layers": 3,
            "vector_size": 128,
            "hidden_dimension": 128,
            "transformer_heads": 4,
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "prob_hidden_dim": 16,
                "suggesion_feature_hidden_dim": 16,
                "embedding_dim": 128,
                "flatten_output": False,
            },
        },
        "tensorboard_log": "./logs/tb/",
    },
}


if __name__ == "__main__":
    train(train_envs, model_envs, PPO)
