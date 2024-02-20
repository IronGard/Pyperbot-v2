import gymnasium as gym
import torch
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DDPG, dqn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
import time
import matplotlib
import pyperbot_v2
import yaml

import configparser
import argparse
import os

#TODO: Add in the config setup and the config reading processes.

#=========================Argument Parser=========================
parser = argparse.ArgumentParser(description = 'Run python file with or without arguments')

#add arguments
parser.add_argument('-f', '--filename', help = "Name of python file to run")
parser.add_argument('-a', '--algo', help = "Name of algorithm to be run")
parser.add_argument('-en', '--env', help = "Simulation environment to be used")
parser.add_argument('-t', '--terrain', help = "Terrain to be used for the simulation")
parser.add_argument('-ep', '--episodes', help = "Number of training iterations")
parser.add_argument('-n', '--num_goals', help = "Number of goals to be used in the simulation")
parser.add_argument('-s', '--seed', help = "Seed for the simulation")
parser.add_argument('-n_args', '--num_args', help = "Number of arguments to be used in the simulation")
parser.add_argument('-eps_clip', '--epsilon_clip', help = "Epsilon clip value for the PPO algorithm")
parser.add_argument('-k', '--k_epochs', help = "Number of epochs for the PPO algorithm")
parser.add_argument('-lr', '--learning_rate', help = "Learning rate for the PPO algorithm")
parser.add_argument('-g', '--gamma', help = "Discount factor for the PPO algorithm")
args = vars(parser.parse_args())

#open and read config parser
config = configparser.ConfigParser()
config['DEFAULT'] = {'filename': 'main.py', 
                        'episodes': 1000,
                        'algo': 'PPO',
                        'env': 'SnakebotEnv',
                        'terrain': 'Plane',
                        'num_goals': 1,
                        'seed': 0}

#check if user entered necessary parameters:
config['USER'] = {k: v for k,v in args.items() if v is not None}

#write entries into the config file
with open('config.yaml', 'w') as configfile:
    config.write(configfile)

#TODO: make sure the main file actually parses the inputs from the argument parser properly and takes it properly.
def main():
    #here, we use the SB3 based implementation of the PPO algorithm - though a custom one could be defined as well.
    env = gym.make("SnakebotEnv-v0")
    #convert env to vector environment
    agent = PPO("MlpPolicy", "SnakebotEnv-v0", verbose = 1)
    print("Environment found. Training agent...")
    agent.learn(total_timesteps = int(25000), progress_bar = True)
    agent.save("PPO_snakebot")
    #reset the environment and render
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes = 10)
    print(f'Mean reward = {mean_reward}')
    print(f'Standard reward = {std_reward}')

    #establishing the vectorised environment. 
    vec_env = agent.get_env()
    obs = vec_env.reset() #call reset at the beginning of an episode
    check_env(vec_env) #check if the environment is valid

    # #sample action and observation
    action = vec_env.action_space.sample()
    print("Sampled action: ", action)
    obs, reward, done, info = vec_env.step(action)

    for i in range(1000):
        action, states = agent.predict(obs)
        obs, rewards, terminated, info = vec_env.step(action)
        vec_env.render('human')
    # #save results for plotting and analysis.

if __name__ == "__main__":
    main()
