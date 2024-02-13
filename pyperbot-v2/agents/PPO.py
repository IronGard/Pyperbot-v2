#general imports
import logger
import os
import pybullet as p
import pybullet_envs
import gymnasium as gym
import argparse

#SB3 Imports
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

#torch imports
import torch
import torch.nn as nn

#rl_zoo3 import
from rl_zoo3.train import train

#define actor value function networks
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    activation_fn=torch.nn.ReLU
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default = os.path.basename(__file__)[:-3], help = "Name of the environment for the snakebot")
    parser.add_argument('--seed', type=int, default = 0, help = "Seed for the environment")
    parser.add_argument('--model', type=str, default = "PPO", help = "Name of the model to use")
    parser.add_argument('--cuda', type = bool, default = True, help = "If enabled, cuda will be used by default.")

#define snakebot_env class from other work
class SnakebotEnv(gym.Env):
    ''' 
    Class for environment for the snake robot. Takes in the following params:

    '''
    def __init__(self):


#loading the model
model = PPO("MlpPolicy", , policy_kwargs=policy_kwargs, verbose=1)