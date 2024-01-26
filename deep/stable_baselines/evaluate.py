import sys

from stable_baselines3 import DQN

from deep.stable_baselines._train import evaluate_model
from pylixir.envs.DictPylixirEnv import DictPylixirEnv

model_zip_path = sys.argv[
    1
]  # ex.  "./logs/checkpoints/DQN.exp-neg-decay-b128-emb/rl_model_1500000_steps.zip"

model = DQN.load(model_zip_path)

env = DictPylixirEnv()
av_ep_lens, avg_rewards, success_rate, r_14, r_16, r_18 = evaluate_model(
    model, env, max_seed=1000, threshold=14, render=False
)
print(
    "--------------------------------------------------------------------------------------------"
)
print("average episode length : ", av_ep_lens)
print("mean of average reward of each episode : ", avg_rewards)
print("success rate (%) : ", success_rate * 100)
print("success rate[14] (%) : ", r_14 * 100)
print("success rate[16] (%) : ", r_16 * 100)
print("success rate[18] (%) : ", r_18 * 100)

print(
    "--------------------------------------------------------------------------------------------"
)
