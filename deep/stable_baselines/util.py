from typing import TypedDict


class TrainSettings(TypedDict):
    name: str
    expname: str
    total_timesteps: int
    log_interval: int  # log avg reward in the interval (in num timesteps)
    checkpoint_freq: int
    eval_freq: int
    evaluation_n: int  # n of episodes to simulate in evaluation phase
    n_envs: int


def get_basic_train_settings(name: str) -> TrainSettings:
    basic_train_setting: TrainSettings = {
        "name": name,
        "expname": "",
        "total_timesteps": int(5e5),
        "log_interval": int(1e3),
        "checkpoint_freq": int(1e5),
        "eval_freq": int(1e5),
        "evaluation_n": int(250),
        "n_envs": 1,
    }
    return basic_train_setting


class ModelSettings(TypedDict):
    policy: str
    learning_rate: float
    seed: float
    kwargs: dict  # network-specific hyperparams
