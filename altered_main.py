#main file changed to accommodate separate testing and training functions.
import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt
import tensorboard
import pyperbot_v2
import pybullet as p
import warnings
import tensorflow as tf
import argparse
import os
import pandas as pd
import numpy as np
import json

#sb3 imports
import stable_baselines3 as sb3
from stable_baselines3 import PPO, TD3, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecNormalize, VecMonitor, SubprocVecEnv #add option to normalise reward + observation
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
# from stable_baselines3.common.preprocessing import NormalizeReward, NormalizeObservation
from stable_baselines3.common import results_plotter

#other relative imports
from pyperbot_v2.custom_agents.PPO import CustomPPO
from pyperbot_v2.wrappers.TimeLimitEnv import TimeLimitEnv
from pyperbot_v2.wrappers.SaveOnStepCallback import SaveOnStepCallback
from pyperbot_v2.utils.utils import file_clear, plot_results, moving_average

#=========================Constants=========================
host = '0.0.0.0'
port = 6969
sample_step = 20
counter = 0
starting_sample = 150
#===========================================================
log_dir = "pyperbot_v2/logs/"
os.makedirs(log_dir, exist_ok = True)
results_dir = "pyperbot_v2/results/"
model_dir = "pyperbot_v2/models/files/"
config_dir = "pyperbot_v2/config/seeded_run_configs/"
params_dir = "pyperbot_v2/models/params/"
os.makedirs(params_dir, exist_ok = True)
#===========================================================

def make_env(env_id, rank, num_envs, seed=0, **kwargs):
    def _init():
        env = gym.make(env_id, num_envs = num_envs, **kwargs)
        env.seed(seed + rank)
        return env
    return _init

def get_action_from_norm(action):
    joint_upper_limits = [0.01, 0.0873, 0.2618, 0.5236, 0.5236, 0.01, 0.0873, 0.2618, 0.5236, 0.5236, 0.01, 0.0873, 0.2618, 0.5236, 0.5236, 0.01, 0.0873, 0.2618, 0.5236, 0.5236]
    joint_lower_limits = [-0.01, -0.0873, -0.2618, -0.5236, -0.5236, -0.01, -0.0873, -0.2618, -0.5236, -0.5236, -0.01, -0.0873, -0.2618, -0.5236, -0.5236, -0.01, -0.0873, -0.2618, -0.5236, -0.5236]
    action = [action[i] * (joint_upper_limits[i] - joint_lower_limits[i])/2 for i in range(len(action))]
    return action

def train(args):
    '''
    Function used to train the model for certain number of timesteps.
    Added customisation for PPO model. 
    '''
    #create environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.vectorise:
        env_kwargs = {'terrain': args.terrain} 
        num_envs = 4  # Number of vectorized environments
        env = SubprocVecEnv([make_env(args.environment, i, num_envs, **env_kwargs) for i in range(num_envs)])
        env = VecMonitor(env, log_dir)
        if args.normalise:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        # eval_callback = EvalCallback(env, best_model_save_path=os.path.join(results_dir, f'{args.rl_algo}', 'best_models'), 
        #                             log_path = os.path.join(results_dir, f'{args.rl_algo}', 'np_plots'), 
        #                             eval_freq = int(args.timesteps)/10, deterministic = True,
        #                             render = False)
    else:
        env = DummyVecEnv([lambda: gym.make(args.environment)])
        env = TimeLimitEnv(env, max_episode_steps = 20000)
        env = VecMonitor(env, log_dir)
        if args.normalise:
            env = VecNormalize(env, norm_obs = True, norm_reward = True, clip_obs = 10.)
        # env = gym.make(args.environment)
        # env = TimeLimitEnv(env, max_episode_steps = 25000)
        # env = Monitor(env, log_dir)
        # if args.normalise:
        #     env = NormalizeReward(env, gamma=0.99)
        #     env = NormalizeObservation(env)
    
    seed_value = env.seed(0)
    obs = env.reset()
    print(f"Simulation seed: {seed_value}")
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(results_dir, f'{args.rl_algo}', 'best_models'), 
                                log_path = os.path.join(results_dir, f'{args.rl_algo}', 'np_plots'), 
                                eval_freq = int(args.timesteps)/10, deterministic = False,
                                render = False)
    
    # custom_callback = SaveOnStepCallback(log_dir = os.path.join(results_dir, f'{args.rl_algo}'))
    # callback_list = [custom_callback, eval_callback]

    #load best model params from optuna hyperparameter tuning.
    #normalise the observations and rewards and clip observations
    if args.rl_algo == "PPO":
        if args.custom:
            agent = CustomPPO(env, learning_rate = args.learning_rate, batch_size = args.batch_size, gamma = args.gamma, clip_epslion = args.clip_epsilon, value_loss_coef = args.value_loss_coef, entropy_coef = args.entropy_coef)
        else:
            agent = PPO("MlpPolicy", env, verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'), device = device)
    elif args.rl_algo == "TD3":
        agent = TD3("MlpPolicy", env, verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'))
    elif args.rl_algo == "DDPG":
        agent = DDPG("MlpPolicy", env, verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'))
    elif args.rl_algo == "DQN":
        agent = DQN("MlpPolicy", env, verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'))
    else:
        raise ValueError("Invalid reinforcement learning algorithm specified. Please specify a valid algorithm.")
    print("Environment found. Training agent...")

    #training the model
    agent.learn(total_timesteps = int(args.timesteps), progress_bar = True, tb_log_name = f"{args.rl_algo}_snakebot_{args.timesteps}ts", callback = eval_callback)
    agent.save(os.path.join(model_dir, f"{args.rl_algo}_snakebot_seed{seed_value}"))
    agent_params = agent.get_parameters()
    agent_params['seed'] = seed_value
    torch.save(agent.policy.state_dict(), os.path.join(params_dir, f'{args.rl_algo}_snakebot_seed{seed_value}.pt'))
    print(f"Model saved successfully as {args.rl_algo}_snakebot_seed{seed_value}.")
    env.close()

def test(args):
    '''
    Function used to load and test models according to the name and seed passed
    by user. 
    '''
    env = gym.make(args.environment)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 20000)
    env = Monitor(env, log_dir)
    if args.normalise:
        env = VecNormalize(env, norm_obs = True, norm_reward = True, clip_obs = 10.)
    if args.rl_algo == "PPO":
        agent = PPO.load(os.path.join(model_dir, f"{args.rl_algo}_snakebot_seed{[int(args.seed)]}.zip"))
    elif args.rl_algo == "TD3":
        agent = TD3.load(os.path.join(model_dir, f"{args.rl_algo}_snakebot_seed{int(args.seed)}.zip"))
    elif args.rl_algo == "DDPG":
        agent = DDPG.load(os.path.join(model_dir, f"{args.rl_algo}_snakebot_seed{int(args.seed)}.zip"))
    elif args.rl_algo == "DQN":
        agent = DQN.load(os.path.join(model_dir, f"{args.rl_algo}_snakebot_seed{int(args.seed)}.zip"))
    else:
        raise ValueError("Invalid reinforcement learning algorithm specified. Please specify a valid algorithm.")
    print("Model loaded successfully. Testing agent...")
    if args.save_action:
        num_episodes = args.episodes
        episode_reward_arr = []
        cum_reward_arr = []
        action_arr = []
        for episode in range(num_episodes):
            obs = np.array(env.reset()[0])
            done = False
            total_reward = 0
            for timestep in range(int(args.timesteps)):
                action, states = agent.predict(obs, deterministic = False)
                obs, reward, done, _, info = env.step(action)
                action = get_action_from_norm(action)
                action_arr.append(action)
                total_reward += reward
                episode_reward_arr.append(reward)
            print(f'Episode: {episode + 1}, Total reward: {total_reward}')
        
        #plot rewards vs timesteps for both cumulative and episode rewards
        plt.plot(episode_reward_arr)
        plt.title("Episode rewards")
        plt.xlabel("Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(os.path.join(results_dir, f'{args.rl_algo}', f'episode_rewards_{args.rl_algo}_snakebot_seed{int(args.seed)}.png'))
        plt.close()
                
        #saving actions to csv
        action_df = pd.DataFrame(action_arr)
        action_df.to_csv(os.path.join(results_dir, f'{args.rl_algo}', f'actions_{args.rl_algo}_snakebot_seed{int(args.seed)}.csv'), index = False)
        
    #establishing mean and std reward
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes = 10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    env.close()

def get_actions_from_model(model, env, timesteps):
    '''
    return joint positions from model predictions in a csv file
    '''
    env = gym.make(args.environment)
    


def main(args):
    if args.file_clear:
        file_clear(config_dir)
        file_clear(os.path.join(log_dir, 'actions'))
        os.makedirs(os.path.join(log_dir, 'actions'), exist_ok = True)
        os.makedirs(os.path.join(config_dir, 'goal_config'), exist_ok = True)
        os.makedirs(os.path.join(config_dir, 'snake_config'), exist_ok = True)
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise ValueError("Invalid mode. Please select either 'train' or 'test'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run python file with or without arguments')
    #add arguments
    parser.add_argument('-e', '--environment', help = "Name of the environment to be used for the simulation", default = "StandardSnakebotEnv")
    parser.add_argument('-s', '--seed', help = "Load a seed for the simulation", default = 0)
    parser.add_argument('-m', '--mode', help = "Mode to be used for the simulation", default = "train")
    parser.add_argument('-fc', '--file_clear', help = "Decide whether to clear the file or not", default = False)
    parser.add_argument('-sb', '--snakebot', help = "Name of the snakebot to be used for the simulation", default = "simulation")
    parser.add_argument('-sm', '--save_model', help = "Decide whether to save the model or not after training", default = False)
    parser.add_argument('-l', '--load_model', help = "Decide whether to load a pre-trained model or not", default = False)
    parser.add_argument('-sa', '--save_action', help = "Determine whether to save the actions from a model as well", default = True)
    parser.add_argument('-nm', '--normalise', help = "Normalise rewards and observations", default = False)
    parser.add_argument('-rm', '--robot_model', help = "Name of the robot model to be used for simulation", default = "snakebot")
    parser.add_argument('-rl', '--rl_algo', help = "Name of reinforvement learning algorithm to be run", default = "PPO")
    parser.add_argument('-tt', '--timesteps', help = "Number of timesteps for the training process", default = 500000)
    parser.add_argument('-nr', '--num_runs', help = "Number of runs for the timestep", default = 5)
    parser.add_argument('-la', '--load_agent', help = "Load a pre-trained agent", default = False)
    parser.add_argument('-t', '--terrain', help = "Terrain to be used for the simulation", default = 'maze')
    parser.add_argument('-ep', '--episodes', help = "Number of training episodes", default = 10)
    parser.add_argument('-n', '--num_goals', help = "Number of goals to be used in the simulation", default = 1)
    parser.add_argument('-v', '--vectorise', help = "Vectorise the environment", default = False)
    
    #parser arguments for custom model
    parser.add_argument('-c', '--custom', help = "Decide whether to use a custom model or not", default = False)
    parser.add_argument('-lr', '--learning_rate', help = "Learning rate for models", default = 0.0003)
    parser.add_argument('-bs', '--batch_size', help = "Batch size for models", default = 64)
    parser.add_argument('-g', '--gamma', help = "Gamma value for models", default = 0.99)
    parser.add_argument('-ce', '--clip_epsilon', help = "Clip epsilon value for models", default = 0.2)
    parser.add_argument('-vlc', '--value_loss_coef', help = "Value loss coefficient for models", default = 0.5)
    parser.add_argument('-ec', '--entropy_coef', help = "Entropy coefficient for models", default = 0.01)

    args = parser.parse_args()
    # conn = setup_connection(s) #TODO - setup timeout condition for connection timeout
    main(args)