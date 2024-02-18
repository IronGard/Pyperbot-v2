import gymnasium as gym
import torch
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DDPG, dqn
import time
from config import config_setup, read_config
import matplotlib
import pyperbot_v2


#TODO: Add in the config setup and the config reading processes.

def main():
    #here, we use the SB3 based implementation of the PPO algorithm - though a custom one could be defined as well.
    env = gym.make("SnakebotEnv-v0")
    #convert env to vector environment
    agent = PPO("MlpPolicy", "SnakebotEnv-v0", verbose = 1)
    print("Environment found. Training agent...")
    agent.learn(total_timesteps = 25000)
    #reset the environment and render
    ob = env.reset()
    env.render()
    while True:
        action = agent(ob)
        reward, obs, done = env.step(action)
        # if done:
        #     ob = env.reset()
        #     time.sleep(1/30)
        #save results for plotting and analysis
        env.render()
        print(reward)
        print(f'Joint positions: {obs}')

if __name__ == "__main__":
    main()
