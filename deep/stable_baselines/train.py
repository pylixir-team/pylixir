import os, time, random
from typing import Type, Union
import json
from pathlib import Path


import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import trange

from deep.stable_baselines.util import ModelSettings, TrainSettings
from pylixir.envs import register_env

ENV_NAME = "DictPylixirEnv"


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose=0, eval_freq=10000, n_eval_episodes=1000):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.eval_freq == 0:
            mean, std, success_rate = evaluate(self.model, self.training_env, max_seed=self.n_eval_episodes)
            self.logger.record("eval/mean", mean)
            self.logger.record("eval/std", std)
            self.logger.record("eval/success_rate", success_rate)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

def _serialize_config(nested_dict):
    if isinstance(nested_dict, (int, str, float, bool)):
        return nested_dict

    if isinstance(nested_dict, list):
        return [
            _serialize_config(vv) for vv in nested_dict
        ]

    output = {}
    for k, v in nested_dict.items():
        if isinstance(v, (list, dict)):
            output[k] = _serialize_config(v)
        elif isinstance(v, (int, str, float, bool)):
            output[k] = v
        else:
            output[k] = str(v)

    return output


def train(
    train_envs: TrainSettings, model_envs: ModelSettings, Model: Type[BaseAlgorithm]
) -> None:
    n_envs = train_envs["n_envs"]
    # Env Control
    register_env()
    env = make_vec_env(f"pylixir/{ENV_NAME}-v0", env_kwargs={"render_mode": "human"}, n_envs=n_envs)
    # env = PylixirEnv()
    # env.reset(0)
    action_dim = env.action_space.n

    # Paint all settings to console
    print("training environment name : " + ENV_NAME)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", train_envs["total_timesteps"])
    print(
        "model saving frequency : " + str(train_envs["checkpoint_freq"]) + " timesteps"
    )
    print("log frequency : " + str(train_envs["log_interval"]) + " timesteps")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("action space dimension : ", action_dim)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("model envs:")
    print(*(model_envs["kwargs"].items()), sep="\n")

    # Create Model
    model = Model(
        model_envs["policy"], env, model_envs["learning_rate"], seed=model_envs["seed"], **model_envs["kwargs"]
    )
    model.set_random_seed(model_envs["seed"])
    checkpoint_callback = get_callback(
        train_envs["checkpoint_freq"] // n_envs, train_envs["eval_freq"] // n_envs, f"./logs/checkpoints/{train_envs['name']}.{train_envs['expname']}"
    )
    model_dirname = f"logs/checkpoints/{train_envs['name']}.{train_envs['expname']}"
    try:
        Path(model_dirname).mkdir(parents=True, exist_ok=False)
    except Exception as e:
        print("Given exp name already exist")
        raise e

    with open(os.path.join(model_dirname, "config.json"), "w") as f:
        json.dump(_serialize_config({
            "train": train_envs,
            "model": model_envs
        }), f, indent=2)

    print(model.policy)
    random.seed(model_envs["seed"])
    evaluate(
        model, env, max_seed=train_envs["evaluation_n"], render=False
    )

    # Train Model
    model.learn(
        train_envs["total_timesteps"],
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=train_envs["log_interval"],
        tb_log_name=f"{train_envs['name']}.{train_envs['expname']}",
    )
    # Save Model
    model_path = f"logs/checkpoints/{train_envs['name']}.{train_envs['expname']}/latest.zip"
    model.save(model_path)
    # model.set_parameters(model_path)

    random.seed(model_envs["seed"])
    evaluate(
        model, env, max_seed=train_envs["evaluation_n"], render=False
    )
    # model.set_parameters(model_path)


def get_callback(
    checkpoint_freq: int, eval_freq: int, checkpoint_path: str
) -> CallbackList:
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, save_path=checkpoint_path # , name_prefix=checkpoint_name
    )
    # checkpoint_callback = EveryNTimesteps(n_steps=checkpoint_freq, callback=checkpoint_callback)
    eval_callback = CustomCallback(eval_freq=eval_freq)
    # eval_callback = EveryNTimesteps(n_steps=eval_freq, callback=eval_callback)
    # callback = CallbackList([checkpoint_callback, eval_callback])
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=checkpoint_path,
    #     log_path=checkpoint_path,
    #     eval_freq=eval_freq,
    #     deterministic=True,
    #     render=False,
    # )
    callback = CallbackList([checkpoint_callback, eval_callback])
    # callback = EveryNTimesteps(n_steps=eval_freq, callback=callback)
    return callback

# def evaluate(
#     model: BaseAlgorithm, env: Union[gym.Env, VecEnv], threshold: int = 14, max_seed: int = 100000, render: bool = False
# ):
#     now = time.time()
#     av_ep_lens, avg_rewards, success_rate = evaluate_model(
#         model, env, threshold=threshold, max_seed=max_seed, render=render
#     )
#     print(f"Time taken: {time.time() - now}")
#     print(
#         "--------------------------------------------------------------------------------------------"
#     )
#     print("average episode length : ", av_ep_lens)
#     print("mean of average reward of each episode : ", avg_rewards)
#     print("success rate (%) : ", success_rate * 100)
#     print(
#         "--------------------------------------------------------------------------------------------"
#     )


def evaluate_model(
    model: BaseAlgorithm, env: Union[gym.Env, VecEnv], threshold: int = 14, max_seed: int = 100000, render: bool = False
) -> tuple[float, float, float]:
    av_ep_lens, avg_rewards, success_rate = 0, 0, 0
    neg_rew = 0
    success_rate_14, success_rate_16, success_rate_18 = 0, 0, 0
    for seed in trange(max_seed):
        obs, _ = env.reset(seed=seed)
        if render:
            env.render()
        terminated = False
        curr_reward, curr_ep_len = 0, 0
        ticks = 0
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, info = env.step(action)
            if render:
                env.render()
            curr_reward += reward
            curr_ep_len += 1
            ticks += 1
            if ticks > 20:
                break

        av_ep_lens += curr_ep_len
        avg_rewards += curr_reward
        if ticks < 20:
            if info["current_valuation"] >= threshold:
                success_rate += 1
            if info["current_valuation"] >= 14:
                success_rate_14 += 1
            if info["current_valuation"] >= 16:
                success_rate_16 += 1
            if info["current_valuation"] >= 18:
                success_rate_18 += 1
        else:
            neg_rew += 1

    print(f"Wrong choice: {neg_rew}")

    return tuple(
        map(lambda x: float(x / max_seed), (av_ep_lens, avg_rewards, success_rate, success_rate_14, success_rate_16, success_rate_18))
    )

def evaluate(
    model: BaseAlgorithm, env: Union[gym.Env, VecEnv], threshold: int = 14, max_seed: int = 100000, render: bool = False
):
    def callback(local_vars, global_vars):
        nonlocal success_rate_14, success_rate_16, success_rate_18
        nonlocal average_enchant_count # or global cnt for global variable cnt
        if local_vars["done"]:
            success_rate_14 += local_vars["info"]["current_valuation"] >= 14
            success_rate_16 += local_vars["info"]["current_valuation"] >= 16
            success_rate_18 += local_vars["info"]["current_valuation"] >= 18
            average_enchant_count += local_vars["info"]["current_valuation"]

    now = time.time()
    success_rate_14 = 0
    success_rate_16 = 0
    success_rate_18 = 0
    average_enchant_count = 0

    # random.seed(37)
    mean, std = evaluate_policy(model, env, n_eval_episodes=max_seed, render=render, callback=callback)
    print(f"mean: {mean}, std: {std}")
    print(f"Success rate[14] (%): {success_rate_14 / max_seed * 100}")
    print(f"Success rate[16] (%): {success_rate_16 / max_seed * 100}")
    print(f"Success rate[18] (%): {success_rate_18 / max_seed * 100}")
    print(f"Average enchant count: {average_enchant_count / max_seed}")
    print(f"Time taken: {time.time() - now}")
    print(f"max_seed: {max_seed}")
    return mean, std, success_rate_14 / max_seed
