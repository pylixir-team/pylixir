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


# Evaluate model: SB3 evaluate_policy version
# No advantage!
# from stable_baselines3.common.evaluation import evaluate_policy
# cnt = 0
# def callback(local_vars, global_vars):
#     nonlocal cnt # or global cnt for global variable cnt
#     if local_vars["done"]:
#         cnt += local_vars["info"]["total_reward"] >= 14
# print(evaluate_policy(model, env, n_eval_episodes=train_envs["evaluation_n"], render=False, callback=callback))
# print(f"Success rate (%): {cnt / train_envs['evaluation_n'] * 100}")