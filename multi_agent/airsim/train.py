import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from scripts.network import CustomCombinedExtractor
from stable_baselines3.common.env_checker import check_env
from pettingzoo.test import parallel_api_test
from stable_baselines3.common.vec_env import VecTransposeImage
from scripts.airsim_env import AirSimDroneEnv, petting_zoo

import os
print("当前工作目录：", os.getcwd())
# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Determine input image shape
image_shape = (84,84,1) if config["train_mode"]=="depth" else (84,84,3)

# parallel_api_test(env_parallel)

env = petting_zoo(        
        ip_address="127.0.0.1", 
        image_shape=image_shape,
        input_mode=config["train_mode"],
        num_drones=config["num_drones"]
        )

#check_env(env)
# Manually add VecTransposeImage
env = VecTransposeImage(env)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor
)

print("==========1 Environment created!!!!==========")
# New model
model = PPO(
    'MultiInputPolicy', 
    env, 
    n_steps=512,
    learning_rate=0.00001, #0.0001, 0.00005, 0.00001
    batch_size=256,
    clip_range=0.1, #0.1
    verbose=1, 
    seed=42,
    device="cuda",
    tensorboard_log="./tb_logs/",
    policy_kwargs=policy_kwargs,
    use_sde= True
)

print("========2 Model created!!!!==========")
# Continual learning to save msgpackrpc error
custom_objects = { 'learning_rate': 0.001 } #0.00005, 0.00001

######## traineed model load ############
# model = PPO.load("saved_policy/best_model", custom_objects = custom_objects, tensorboard_log="tb_logs\ppo_run_map1_1696308625.3478022_1", env=env)
######## traineed model load ############

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor
)

################### New model setup###################
print('==========================================================')
print('Model Design:')
print(model.policy)
print('==========================================================')

# Evaluation callback
callbacks = []
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=230, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=callback_on_best,
    n_eval_episodes=30,
    best_model_save_path="saved_policy",
    #log_path=".", # Not for load model
    eval_freq=8192,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "ppo_run_map1_1696308625.3478022_1" #+ str(time.time())


model.learn(
    total_timesteps=20000,
    tb_log_name=log_name,
    reset_num_timesteps = False, # For load model
    **kwargs
)
