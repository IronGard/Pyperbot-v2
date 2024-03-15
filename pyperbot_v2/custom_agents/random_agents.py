import torch
import stable_baselines3 as sb3
import gymnasium as gym
import os
import pybullet as p

#designing an agent that takes random actions for the snake robot.

class RandomAgent:
    def __init__(self, env):
        self.env = env
    
    def act(self, obs):
        return self.env.action_space.sample()

    def train(self, num_episodes=100):
        reward = 0
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_obs, reward, done, _ = self.env.step(action)
                
                obs = next_obs

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, env):
        return cls(env)