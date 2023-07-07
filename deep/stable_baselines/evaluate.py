from stable_baselines3 import DQN

from deep.stable_baselines.train import evaluate_model
from pylixir.envs.PylixirEnv import PylixirEnv

model = DQN.load("./model/DQN/DQN_2.zip")
env = PylixirEnv()
av_ep_lens, avg_rewards, success_rate = evaluate_model(model, env, max_seed=int(10e3))
print(
    "--------------------------------------------------------------------------------------------"
)
print("average episode length : ", av_ep_lens)
print("mean of average reward of each episode : ", avg_rewards)
print("success rate (%) : ", success_rate * 100)
print(
    "--------------------------------------------------------------------------------------------"
)
