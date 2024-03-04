import gymnasium as gym
import torch
import time
import matplotlib
import matplotlib.pyplot as plt
import tensorboard
import pyperbot_v2
import pybullet as p
import warnings
import tensorflow as tf
import argparse
import os
import pandas as pd

#sb3 imports
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

#suppress warnings
warnings.filterwarnings("ignore")

#making relative imports from server_code.py
from pyperbot_v2.wrappers.TimeLimitEnv import TimeLimitEnv

#=========================Constants=========================
host = '0.0.0.0'
port = 6969
sample_step = 20
counter = 0
starting_sample = 150
#===========================================================

#TODO: Add in the config setup and the config reading processes.
#setting up logging directory in tensorboard
log_dir = "pyperbot_v2/tensorboard_logs/ppo/"
results_dir = "pyperbot_v2/results/"
model_dir = "pyperbot_v2/models/"
#===========================================================

#TODO: make sure the main file actually parses the inputs from the argument parser properly and takes it properly.
def main(env, rl_algo, timesteps, num_runs, load_agent, terrain, episodes, num_goals, seed, k_epochs, learning_rate, gamma):
    #here, we use the SB3 based implementation of the PPO algorithm - though a custom one could be defined as well.
    #env = gym.make("SnakebotEnv-v0")
    env = Monitor(gym.make(str(env)))
    # env = Monitor(gym.make("LoaderSnakebotEnv-v0"))
    # env = TimeLimitEnv(env, max_episode_steps = 1000) #enforce maximum step limit on agent to find goals
    #convert env to vector environment
    if rl_algo == "PPO":
        agent = PPO("MlpPolicy", "ModSnakebotEnv-v0", verbose = 1, tensorboard_log = log_dir)
    elif rl_algo == "A2C":
        agent = A2C("MlpPolicy", "ModSnakebotEnv-v0", verbose = 1, tensorboard_log = log_dir)
    elif rl_algo == "DDPG":
        agent = DDPG("MlpPolicy", "ModSnakebotEnv-v0", verbose = 1, tensorboard_log = log_dir)
    elif rl_algo == "DQN":
        agent = DQN("MlpPolicy", "ModSnakebotEnv-v0", verbose = 1, tensorboard_log = log_dir)
    else:
        raise ValueError("Invalid reinforcement learning algorithm specified. Please specify a valid algorithm.")
    print("Environment found. Training agent...")
    agent.learn(total_timesteps = int(timesteps), progress_bar = True, tb_log_name = "PPO_snakebot")
    agent.save(os.path.join(model_dir, "PPO_snakebot"))
    #reset the environment and render
    # mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes = num_runs, )
    # print(f'Mean reward = {mean_reward}')
    # print(f'Standard reward = {std_reward}')

    #establishing the vectorised environment. 
    vec_env = make_vec_env("ModSnakebotEnv-v0", n_envs = 4)
    obs = vec_env.reset() #call reset at the beginning of an episode

    # #sample action and observation
    action = env.action_space.sample()
    print("Sampled action: ", action)
    env.reset()
    
    obs, reward, done, _, info = env.step(action)
    position_arr = []
    joint_position_arr = []

    for i in range(1000):
        action, states = agent.predict(obs)
        env.render()
        print(f'Action {i} = {action}')
        obs, rewards, terminated, _, info = env.step(action)
        print(f'Observation {i} = {obs}')
        print(f'Reward {i} = {rewards}')
        position_arr.append(obs[0])
        joint_position_arr.append(action)
        time.sleep(1./240.)
        if done: 
            break
        #TODO - setup live data transfer instead of reading results from CSV

    #generate plot for position array
    plt.plot(position_arr)
    plt.title('Position of the snakebot')
    plt.xlabel('Time step')
    plt.ylabel('Position')
    plt.legend(['x', 'y', 'z'])
    plt.savefig(os.path.join(results_dir, 'csv', 'position_plot.png'))
    plt.show()

    #save positions and joint velocities to csv files
    joint_positions_df = pd.DataFrame(joint_position_arr)
    joint_positions_df.to_csv(os.path.join(results_dir, 'csv', 'joint_positions.csv')) 
    #figure out how to save the position array to the correct address

    #generate plot for the rewards
    matplotlib.use('Agg')
    plt.plot(rewards)
    plt.savefig('rewards_plot.png')
    plt.show()

    #generate plot for loss against episodes
    plt.plot(agent.ep_info_buffer)
    plt.savefig('loss_plot.png')
    plt.show()

    # evaluate model
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes = 10)
    print(f'Mean reward = {mean_reward}')
    print(f'Standard reward = {std_reward}')
    # #save results for plotting and analysis.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run python file with or without arguments')
    #add arguments
    parser.add_argument('-e', '--env', help = "Name of the environment to be used for the simulation", default = "ModSnakebotEnv-v0")
    parser.add_argument('-rl', '--rl_algo', help = "Name of reinforvement learning algorithm to be run", default = "PPO")
    parser.add_argument('-tt', '--timesteps', help = "Number of timesteps for the training process", default = 25000)
    parser.add_argument('-nr', '--num_runs', help = "Number of runs for the timestep", default = 5)
    parser.add_argument('-la', '--load_agent', help = "Load a pre-trained agent", default = False)
    parser.add_argument('-t', '--terrain', help = "Terrain to be used for the simulation", default = 'maze')
    parser.add_argument('-ep', '--episodes', help = "Number of training iterations", default = 1000)
    parser.add_argument('-n', '--num_goals', help = "Number of goals to be used in the simulation", default = 3)
    parser.add_argument('-s', '--seed', help = "Seed for the simulation", default = 42)
    parser.add_argument('-k', '--k_epochs', help = "Number of epochs for the PPO algorithm", default = 4)
    parser.add_argument('-lr', '--learning_rate', help = "Learning rate for the PPO algorithm", default = 0.0003)
    parser.add_argument('-g', '--gamma', help = "Discount factor for the PPO algorithm", default = 0.99)
    args = vars(parser.parse_args())
    print(args)
    # conn = setup_connection(s) #TODO - setup timeout condition for connection timeout
    main(args['env'],
         args['rl_algo'],
         args['timesteps'], 
         args['num_runs'], 
         args['load_agent'], 
         args['terrain'], 
         args['episodes'], 
         args['num_goals'], 
         args['seed'], 
         args['k_epochs'], 
         args['learning_rate'], 
         args['gamma'])
