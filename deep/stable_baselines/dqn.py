from deep.stable_baselines.train import train

train_envs = {
    'name': 'DQN',
    'random_seed': 0,                       # set random seed if required (0 = no random seed)
    'max_ep_len': 20,
    'max_training_timesteps': int(10e6),
    'print_freq': 30000,                    # print avg reward in the interval (in num timesteps)
    'log_freq': 2000,                       # log avg reward in the interval (in num timesteps)
    'save_model_freq': int(5e5),            # save model frequency (in num timesteps)
    'run_num_pretrained': 0,                # change this to prevent overwriting weights in same env_name folder
}

model_envs = {

}

train(train_envs, model_envs)