import gymnasium as gym
import torch
import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DDPG, dqn
import time
from config import config_setup, read_config

def main():
    agent = sb3.PPO("MlpPolicy", "SnakeBotEnv", verbose = 1)
    .train("PPO", seed = 0, batch_size = 64, iterations = 100, max_episode_length = 1000, verbose = True)
    env = gym.make("SnakebotEnv-v0")
    ob = env.reset()
    while True:
        action = agent(ob)
        ob, reward, done, _ = env.step(action)
        if done:
            ob = env.reset()
            time.sleep(1/30)

if __name__ == "__main__":
    main()