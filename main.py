# import gym
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
import json

#check gym installation
print(gym.__version__)

#sb3 imports
import stable_baselines3 as sb3
from stable_baselines3 import PPO, TD3, DDPG, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize #add option to normalise reward + observation
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common import results_plotter

#suppress warnings
warnings.filterwarnings("ignore")

#making relative imports from server_code.py
from pyperbot_v2.wrappers.TimeLimitEnv import TimeLimitEnv, SaveOnStepCallback
from pyperbot_v2.utils.utils import file_clear, plot_results, moving_average
#=========================Constants=========================
host = '0.0.0.0'
port = 6969
sample_step = 20
counter = 0
starting_sample = 150
#===========================================================

#TODO: Add in the config setup and the config reading processes.
#setting up logging directory in tensorboard
log_dir = "pyperbot_v2/logs/"
os.makedirs(log_dir, exist_ok = True)
results_dir = "pyperbot_v2/results/"
model_dir = "pyperbot_v2/models/files/"
config_dir = "pyperbot_v2/config/seeded_run_configs/"
params_dir = "pyperbot_v2/models/params/"
#===========================================================

def get_action_from_norm(action):
    joint_upper_limits = [0.01, 0.0873, 0.2618, 0.5236, 0.5236, 0.01, 0.0873, 0.2618, 0.5236, 0.5236, 0.01, 0.0873, 0.2618, 0.5236, 0.5236, 0.01, 0.0873, 0.2618, 0.5236, 0.5236]
    joint_lower_limits = [-0.01, -0.0873, -0.2618, -0.5236, -0.5236, -0.01, -0.0873, -0.2618, -0.5236, -0.5236, -0.01, -0.0873, -0.2618, -0.5236, -0.5236, -0.01, -0.0873, -0.2618, -0.5236, -0.5236]
    action = [action[i] * (joint_upper_limits[i] - joint_lower_limits[i])/2 for i in range(len(action))]
    return action


#TODO: make sure the main file actually parses the inputs from the argument parser properly and takes it properly.
def main(args):
    #here, we use the SB3 based implementation of the PPO algorithm - though a custom one could be defined as well.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device found: {device}")
    env = gym.make(args.environment)
    seed = env.seed()
    print(f'Simulation seed: {seed}')
    # env = gym.make(f'pyperbot_v2/{args.environment}')
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 2000)
    env = Monitor(env, os.path.join(results_dir, f'{args.rl_algo}'), allow_early_resets = True)
    #create evaluation callback
    eval_callback = EvalCallback(env, best_model_save_path=os.path.join(results_dir, f'{args.rl_algo}', 'best_models'), 
                                 log_path = os.path.join(results_dir, f'{args.rl_algo}', 'np_plots'), 
                                 eval_freq = int(args.timesteps)/10, deterministic = True,
                                 render = False)
    #clear folder
    # file_clear(os.path.join(results_dir, 'csv'))
    # file_clear(os.path.join(results_dir, 'plots'))
    custom_callback = SaveOnStepCallback(log_dir = os.path.join(results_dir, f'{args.rl_algo}'))
    callback_list = [custom_callback, eval_callback]

    if args.normalise:
        env = VecNormalize(env, norm_obs = True, norm_reward = True, clip_obs = 10.)
    if args.rl_algo == "PPO":
        agent = PPO("MlpPolicy", str(args.environment), verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'), device = device, seed = args.seed)
    elif args.rl_algo == "TD3":
        agent = TD3("MlpPolicy", str(args.environment), verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'), device = device, seed = args.seed)
    elif args.rl_algo == "DDPG":
        agent = DDPG("MlpPolicy", str(args.environment), verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'), device = device, seed = args.seed)
    elif args.rl_algo == "DQN":
        agent = DQN("MlpPolicy", str(args.environment), verbose = 1, tensorboard_log = os.path.join(results_dir, f'{args.rl_algo}'), device = device, seed = args.seed)
    else:
        raise ValueError("Invalid reinforcement learning algorithm specified. Please specify a valid algorithm.")
    print("Environment found. Training agent...")
    agent_params = agent.get_parameters()
    agent_params['seed'] = args.seed
    with open(os.path.join(params_dir, f'{args.rl_algo}_params_seed{args.seed}.json'), 'w') as f:
        json.dump(agent_params, f, indent = 4)
    
    #training the model
    agent.learn(total_timesteps = int(args.timesteps), progress_bar = True, tb_log_name = f"{args.rl_algo}_snakebot_{args.timesteps}ts", callback = callback_list)
    agent.save(os.path.join(model_dir, f"{args.rl_algo}_snakebot_seed{args.seed}"))

    # Vectorise environment
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize.load(os.path.join(model_dir, f"{args.rl_algo}_snakebot"), env)

    #load best model for training and eval
    # agent = PPO.load(os.path.join(model_dir, f"{args.rl_algo}_snakebot"), env = env)

    # #sample action and observation
    action = env.action_space.sample()
    print("Sampled action: ", get_action_from_norm(action))
    env.reset()
    
    obs, reward, done, info = env.step(action)
    position_arr = []
    joint_position_arr = []

    counter = 0
    while True:
        action, states = agent.predict(obs)
        env.render()
        print(f'Action {counter} = {get_action_from_norm(action)}')
        obs, rewards, terminated, info = env.step(action)
        print(f'Observation {counter} = {obs}')
        print(f'Reward {counter} = {rewards}')
        position_arr.append(obs[0])
        joint_position_arr.append(get_action_from_norm(action))
        time.sleep(1./240.)
        counter += 1
        if done: 
            print("Done condition achieved!")
            break
        if counter >= 1998:
            print("Counter limit reached!")
            break
        #TODO - setup live data transfer instead of reading results from CSV

    # #generate plot for position array
    # plt.plot(position_arr)
    # plt.title('Position of the snakebot')
    # plt.xlabel('Time step')
    # plt.ylabel('Position')
    # plt.legend(['x', 'y', 'z'])
    # plt.savefig(os.path.join(results_dir, 'plots', 'position_plot.png'))
    # plt.show()

    #save positions and joint velocities to csv files
    joint_positions_df = pd.DataFrame(joint_position_arr)
    joint_positions_df.to_csv(os.path.join(results_dir, f'{args.rl_algo}', 'csv', f'seed{int(seed[0])}_joint_positions.csv')) 
    #figure out how to save the position array to the correct address

    #plot results
    results_plotter.plot_results([os.path.join(results_dir, f'{args.rl_algo}', 'csv', f'seed{int(seed[0])}_joint_positions.csv')], 
                                 int(args.timesteps),
                                 results_plotter.X_TIMESTEPS,
                                 f'{args.rl_algo} Snakebot')
    
    #plot results against episodes
    results_plotter.plot_results([os.path.join(results_dir, f'{args.rl_algo}', 'csv', f'seed{int(seed[0])}_joint_positions.csv')],
                                  int(args.episodes),
                                  results_plotter.X_EPISODES,
                                  f'{args.rl_algo} Snakebot')
    
    #using plot results instead
    # plot_results(f'log_dir_{args.rl_algo}', f'{args.rl_algo} Snakebot')
    # plt.savefig(os.path.join(results_dir, 'plots', f'{args.rl_algo}_snakebot_plot.png'))

    # evaluate model
    mean_reward, std_reward = evaluate_policy(agent, agent.get_env(), n_eval_episodes = 10, callback = eval_callback)
    print(f'Mean reward = {mean_reward}')
    print(f'Standard reward = {std_reward}')
    # #save results for plotting and analysis.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run python file with or without arguments')
    #add arguments
    parser.add_argument('-e', '--environment', help = "Name of the environment to be used for the simulation", default = "LoaderSnakebotEnv")
    parser.add_argument('-s', '--seed', help = "Load a seed for the simulation", default = 0)
    parser.add_argument('-sb', '--snakebot', help = "Name of the snakebot to be used for the simulation", default = "simulation")
    parser.add_argument('-sm', '--save_model', help = "Decide whether to save the model or not after training", default = False)
    parser.add_argument('-l', '--load_model', help = "Decide whether to load a pre-trained model or not", default = False)
    parser.add_argument('-nm', '--normalise', help = "Normalise rewards and observations", default = False)
    parser.add_argument('-rm', '--robot_model', help = "Name of the robot model to be used for simulation", default = "snakebot")
    parser.add_argument('-rl', '--rl_algo', help = "Name of reinforvement learning algorithm to be run", default = "PPO")
    parser.add_argument('-tt', '--timesteps', help = "Number of timesteps for the training process", default = 25000)
    parser.add_argument('-nr', '--num_runs', help = "Number of runs for the timestep", default = 5)
    parser.add_argument('-la', '--load_agent', help = "Load a pre-trained agent", default = False)
    parser.add_argument('-t', '--terrain', help = "Terrain to be used for the simulation", default = 'maze')
    parser.add_argument('-ep', '--episodes', help = "Number of training iterations", default = 10)
    parser.add_argument('-n', '--num_goals', help = "Number of goals to be used in the simulation", default = 3)
    args = parser.parse_args()
    # conn = setup_connection(s) #TODO - setup timeout condition for connection timeout
    main(args)
