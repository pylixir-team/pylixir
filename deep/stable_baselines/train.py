import os
from pylixir.envs.PylixirEnv import PylixirEnv

ENV_NAME = "Pylixir"


def train(train_envs:dict, model_envs:dict):
    
    # Env Control
    env = PylixirEnv()
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Logging
    log_f_name = create_log_file(**train_envs)

    # Checkpointing
    checkpoint_path = create_checkpoint_directory(**train_envs)
    
    # Painting Settings
    print("training environment name : " + ENV_NAME)
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", train_envs["max_training_timesteps"])
    print("max timesteps per episode : ", train_envs["max_ep_len"])
    print("model saving frequency : " + str(train_envs["save_model_freq"]) + " timesteps")
    print("log frequency : " + str(train_envs["log_freq"]) + " timesteps")
    print("printing average reward over episodes in last : " + str(train_envs["print_freq"]) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")

def create_log_file(name:str, **kwargs) -> str:
    #### log files for multiple runs are NOT overwritten
    log_dir = f"{name}_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, ENV_NAME)
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    #### create new log file for each run
    log_f_name = os.path.join(log_dir, f'{name}_{ENV_NAME}_log_{run_num}.csv')
    print("current logging run number for " + ENV_NAME + " : ", run_num)
    print("logging at : " + log_f_name)
    return log_f_name

def create_checkpoint_directory(name:str, run_num_pretrained:int, random_seed:int, **kwargs) -> str:
    directory = f"{name}_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)
    directory = os.path.join(directory, ENV_NAME)
    if not os.path.exists(directory):
          os.makedirs(directory)
    checkpoint_path = os.path.join(directory, f"PPO_{ENV_NAME}_{random_seed}_{run_num_pretrained}.pth")
    print("save checkpoint path : " + checkpoint_path)
    return checkpoint_path