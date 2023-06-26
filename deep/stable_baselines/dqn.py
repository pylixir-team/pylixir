from deep.stable_baselines.train import train
from stable_baselines3 import DQN

train_envs = {
    'name': 'DQN',
    'random_seed': 0,                       # set random seed if required (0 = no random seed)
    'total_timesteps': int(10e4),
    'print_freq': 30000,                    # print avg reward in the interval (in num timesteps)
    'log_interval': 2000,                   # log avg reward in the interval (in num timesteps)
    'save_model_freq': int(5e4),            # save model frequency (in num timesteps)
    'run_num_pretrained': 0,                # change this to prevent overwriting weights in same env_name folder
}

model_envs = {
    'policy': 'MlpPolicy'
}

if __name__ == '__main__':
    train(train_envs, model_envs, DQN)