import os
from datetime import datetime
from typing import Mapping, Union, Type

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.logger import configure

from pylixir.envs.PylixirEnv import PylixirEnv
from deep.stable_baselines.util import TrainSettings, ModelSettings

ENV_NAME = "Pylixir"


def train(
    train_envs: TrainSettings, model_envs: ModelSettings, Model: Type[BaseAlgorithm]
) -> None:

    # Env Control
    env = PylixirEnv()
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Logging
    log_path = create_log_file(train_envs["name"])

    # Checkpointing
    checkpoint_path, checkpoint_name = create_checkpoint_directory(
        train_envs["name"], train_envs["run_num_pretrained"], train_envs["random_seed"]
    )

    # Saving
    model_path = create_model_directory(train_envs["name"])

    # Paint all settings to console
    print("training environment name : " + ENV_NAME)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", train_envs["total_timesteps"])
    print(
        "model saving frequency : " + str(train_envs["save_model_freq"]) + " timesteps"
    )
    print("log frequency : " + str(train_envs["log_interval"]) + " timesteps")
    print(
        "printing average reward over episodes in last : "
        + str(train_envs["print_freq"])
        + " timesteps"
    )
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    ## TODO: paint model params too

    # Create Model
    model = Model(model_envs["policy"], env, model_envs["learning_rate"], verbose=1)
    new_logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(new_logger)
    checkpoint_callback = get_checkpoint_callback(
        train_envs["save_model_freq"], checkpoint_path, checkpoint_name
    )
    model.learn(
        train_envs["total_timesteps"],
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=train_envs["log_interval"],
    )
    model.save(model_path)


def create_log_file(name: str) -> str:
    #### log files for multiple runs are NOT overwritten
    log_path = f"log/logs/{name}_logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print("logging at : " + log_path)
    return log_path


def create_checkpoint_directory(
    name: str, run_num_pretrained: int, random_seed: int
) -> tuple[str, str]:
    checkpoint_path = f"log/checkpoints/{name}_checkpoints"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_name = f"{name}_{random_seed}_{run_num_pretrained}"
    print("save checkpoint path : " + checkpoint_path)
    return checkpoint_path, checkpoint_name


def create_model_directory(name: str) -> str:
    model_path = f"model/{name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    print("save model path : " + model_path)
    return model_path


def get_checkpoint_callback(
    checkpoint_freq: int, checkpoint_path: str, checkpoint_name: str
) -> EveryNTimesteps:
    callback = CheckpointCallback(
        save_freq=1, save_path=checkpoint_path, name_prefix=checkpoint_name
    )
    return EveryNTimesteps(n_steps=checkpoint_freq, callback=callback)
