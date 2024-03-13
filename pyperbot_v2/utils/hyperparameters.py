#Hyperparameter optimisation for the RL model
import gymnasium as gym
import optuna
import os
import sys
import numpy as np
import argparse
import torch

#stable baselines import
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

#adding path to pyperbot_v2
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

#relative imports for environment and model
from custom_agents.PPO import CustomPPO

#Add custom PPO model and ask for hyperparams to optimise reward
def objective(trial):

    #tune four main parameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    clip_epsilon = trial.suggest_uniform('clip_epsilon', 0.1, 0.3)

    #create the environment
    env = gym.make(str(args.environment))   
    env = gym.wrappers.TimeLimit(env, max_episode_steps = args.timesteps)
    env = Monitor(env, 'pyperbot_v2/logs/', allow_early_resets = True)

    #create model
    model = CustomPPO(env, learning_rate, batch_size, gamma, clip_epsilon)
    model.learn(total_timesteps = 1e05)

    #model evaluation
    reward = evaluate_policy(model, env, n_eval_episodes = 10)
    return reward
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hyperparameter optimisation for snakebot")
    parser.add_argument('-e', '--environment', help = "Environment to train the model on", default = 'LoaderSnakebotEnv')
    parser.add_argument('-tt', '--timesteps', help = "Number of timesteps for the simulation", default = 2400)
    args = parser.parse_args()
    study = optuna.create_study(direction = "maximize")
    study.optimize(objective, n_trials = 50)



