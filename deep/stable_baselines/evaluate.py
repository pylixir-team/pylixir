from stable_baselines3 import DQN, PPO

from deep.stable_baselines.train import evaluate_model
from pylixir.envs.DictPylixirEnv import DictPylixirEnv

model = DQN.load("./logs/checkpoints/DQN.exp-neg-decay-b128-emb/rl_model_1500000_steps.zip")
#model = PPO.load("./logs/checkpoints/PPO.init-3e-4/rl_model_1500000_steps.zip")
env = DictPylixirEnv()
av_ep_lens, avg_rewards, success_rate, r_14, r_16, r_18 = evaluate_model(model, env, max_seed=10000, threshold=14, render=False)
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