import os
from typing import Type

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.logger import configure
from tqdm import trange

from deep.stable_baselines.util import ModelSettings, TrainSettings
from pylixir.envs.PylixirEnv import PylixirEnv

ENV_NAME = "Pylixir"


def train(
    train_envs: TrainSettings, model_envs: ModelSettings, Model: Type[BaseAlgorithm]
) -> None:

    # Env Control
    env = PylixirEnv()
    env.reset(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Logging
    log_path = create_log_file(train_envs["name"], train_envs["run_num_pretrained"])

    # Checkpointing
    checkpoint_path, checkpoint_name = create_checkpoint_directory(
        train_envs["name"], train_envs["run_num_pretrained"], train_envs["random_seed"]
    )

    # Saving
    model_path = create_model_directory(
        train_envs["name"], train_envs["run_num_pretrained"]
    )

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
    print("model envs:")
    print(*(model_envs["kwargs"].items()), sep="\n")

    # Create Model
    model = Model(
        model_envs["policy"], env, model_envs["learning_rate"], **(model_envs["kwargs"])
    )
    new_logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(new_logger)
    checkpoint_callback = get_checkpoint_callback(
        train_envs["save_model_freq"], checkpoint_path, checkpoint_name
    )
    print(model.policy)
    # Train Model
    model.learn(
        train_envs["total_timesteps"],
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=train_envs["log_interval"],
    )
    # Save Model
    model.save(model_path)
    # model_path = "log/checkpoints/DQN_checkpoints/DQN_0_2_500000_steps.zip"
    # model = model.load(model_path)
    # Evaluate Model
    av_ep_lens, avg_rewards, success_rate = evaluate_model(
        model, env, max_seed=train_envs["evaluation_n"], render=True
    )
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("average episode length : ", av_ep_lens)
    print("mean of average reward of each episode : ", avg_rewards)
    print("success rate (%) : ", success_rate * 100)
    print(
        "--------------------------------------------------------------------------------------------"
    )


def create_log_file(name: str, run_num_pretrained: int) -> str:
    #### log files for multiple runs are NOT overwritten
    log_path = f"log/logs/{name}_{run_num_pretrained}__logs"
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


def create_model_directory(name: str, run_num_pretrained: int) -> str:
    model_path = f"model/{name}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, f"{name}_{run_num_pretrained}")
    print("save model path : " + model_path)
    return model_path


def get_checkpoint_callback(
    checkpoint_freq: int, checkpoint_path: str, checkpoint_name: str
) -> EveryNTimesteps:
    callback = CheckpointCallback(
        save_freq=1, save_path=checkpoint_path, name_prefix=checkpoint_name
    )
    return EveryNTimesteps(n_steps=checkpoint_freq, callback=callback)


def evaluate_model(
    model: BaseAlgorithm, env: PylixirEnv, threshold: int = 14, max_seed: int = 100000, render: bool = False
) -> tuple[float, float, float]:
    av_ep_lens, avg_rewards, success_rate = 0, 0, 0
    for seed in trange(max_seed):
        obs, _ = env.reset(seed=seed)
        env.render()
        terminated = False
        curr_reward, curr_ep_len = 0, 0
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)
            env.render()
            curr_reward += reward
            curr_ep_len += 1
        av_ep_lens += curr_ep_len
        avg_rewards += curr_reward
        if info["total_reward"] >= threshold:
            success_rate += 1
    return tuple(
        map(lambda x: float(x / max_seed), (av_ep_lens, avg_rewards, success_rate))
    )
