#general imports
import logger
import os
import pybullet as p
import pybullet_envs
import gymnasium as gym
import numpy as np
from gymnasium import spaces, register
import argparse


#SB3 Imports
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

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
    '''
    Defines argument parser for taking in appropriate training models for the PPO algorithm
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default = os.path.basename(__file__)[:-3], help = "Name of the environment for the snakebot")
    parser.add_argument('--seed', type=int, default = 0, help = "Seed for the environment")
    parser.add_argument('--model', type=str, default = "PPO", help = "Name of the model to use")
    parser.add_argument('--cuda', type = bool, default = True, help = "If enabled, cuda will be used by default.")

#Define four cardinal directions for snake robot
LEFT = 0 #left direction
RIGHT = 1 #right direction
FORWARD = 2 #forward direction
BACKWARD = 3 #backward direction

#define snakebot_env class from other work
class SnakebotEnv(gym.Env):
    ''' 
    Class for environment for the snake robot.
    '''
    def __init__(self, grid_dimensions = [200, 200], render_mode = 'GUI'):
        '''
        Constructor for Snakebot Env Class
        '''
        super(SnakebotEnv, self).__init__()
        self.render_mode = render_mode
        self.agent_pos = [0,0]
        self.agent_dir = FORWARD
        self.grid_dimensions = grid_dimensions
        self.action_space = spaces.Discrete(4) #Go forward, back, left and right -> four actions
        self.observation_space = spaces.Box(
            low = np.array([0,0]), high = np.array([self.grid_dimensions[0], self.grid_dimensions[1]]),
            dtype = np.float32
        )
    def step(self, action):
        if action == self.LEFT:
            self.agent_pos = [self.agent_pos[0] - 1, self.agent_pos[1]]
        if action == self.RIGHT: 
            self.agent_pos = [self.agent_pos[0] + 1, self.agent_pos[1]]
        if action == self.FORWARD:
            self.agent_pos = [self.agent_pos[0], self.agent_pos[1] + 1]
        if action == self.BACKWARD:
            self.agent_pos = [self.agent_pos[0], self.agent_pos[1] - 1]
        else:
            raise ValueError(
                'Action should be an integer between 0 and 3, inclusive'
            )
    def seed(self, seed = None):
        pass
    def reset(self, seed = None, options = None):
        super.reset(seed = seed, options = options)
        self.agent_pos = [0,0]
        self.agent_dir = FORWARD
        return np.array([self.agent_pos])
    def render(self, mode = 'GUI'):
        if self.render_mode = 
    def close(self):
        pass
    
