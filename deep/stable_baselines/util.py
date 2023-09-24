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
