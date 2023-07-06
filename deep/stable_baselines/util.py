from abc import ABCMeta
from typing import TypedDict


class TrainSettings(TypedDict):
    name: str
    random_seed: int  # set random seed if required (0 = no random seed)
    total_timesteps: int
    print_freq: int  # print avg reward in the interval (in num timesteps)
    log_interval: int  # log avg reward in the interval (in num timesteps)
    save_model_freq: int  # save model frequency (in num timesteps)
    run_num_pretrained: int  # change this to prevent overwriting weights in same env_name folder
    evaluation_n: int # n of episodes to simulate in evaluation phase

def get_basic_train_settings(name: str) -> TrainSettings:
    basic_train_setting: TrainSettings = {
        "name": name,
        "random_seed": 0,
        "total_timesteps": int(10e5),
        "print_freq": int(10e2),
        "log_interval": int(10e2),
        "save_model_freq": int(10e4),
        "run_num_pretrained": 0,
        "evaluation_n": int(10e3),
    }
    return basic_train_setting


class ModelSettings(TypedDict):
    policy: str
    learning_rate: float
