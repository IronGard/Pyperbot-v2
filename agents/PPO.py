import logger
import os
import pybullet
import pybullet_envs

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from rl_zoo3.train import train


class PPO_model:
    def __init__(self, env_name):
        self.env_name == env_name

    def train(self, snake_length, timesteps, envs, seed, logger, eval, save, render, net_arch):
        