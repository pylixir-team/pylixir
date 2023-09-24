import sys
from copy import deepcopy

import yaml
from stable_baselines3 import DQN, PPO

from deep.stable_baselines._train import train
from deep.stable_baselines.policy.council_feature import CustomCombinedExtractor
from deep.stable_baselines.policy.transformer_network import TransformerQPolicy
from deep.stable_baselines.util import LearningRateDecay, ModelSettings

_POLICY = {
    "TransformerQPolicy": TransformerQPolicy,
}

_FEATURE_EXTRACTOR = {
    "CustomCombinedExtractor": CustomCombinedExtractor,
}

_ARCHITECTURE = {
    "DQN": DQN,
    "PPO": PPO,
}


def as_model_envs(raw_model_env: dict) -> ModelSettings:
    raw_config = deepcopy(raw_model_env)

    # Inject policy
    if raw_config["policy"] in _POLICY:
        raw_config["policy"] = _POLICY[raw_config["policy"]]

    # Inject scheduled learning rate
    if isinstance(raw_config["learning_rate"], dict):
        raw_config["learning_rate"] = LearningRateDecay(
            raw_config["learning_rate"]["start"], raw_config["learning_rate"]["end"]
        )

    # inject feature extractor
    if (
        raw_config["kwargs"]["policy_kwargs"]["features_extractor_class"]
        in _FEATURE_EXTRACTOR
    ):
        raw_config["kwargs"]["policy_kwargs"][
            "features_extractor_class"
        ] = _FEATURE_EXTRACTOR[
            raw_config["kwargs"]["policy_kwargs"]["features_extractor_class"]
        ]

    return raw_config


def start_train(file_path: str) -> ModelSettings:
    with open(file_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_env = config["train"]
    model_env = as_model_envs(config["model"])
    architecture = _ARCHITECTURE[config["architecture"]]

    train(train_env, model_env, architecture)


if __name__ == "__main__":
    start_train(sys.argv[1])
