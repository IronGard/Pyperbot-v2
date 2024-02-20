#general imports
import logger
import os
import pybullet as p
import pybullet_envs
import gymnasium as gym
import numpy as np
from gymnasium import spaces, register
import argparse


#torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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
    vars = parser.parse_args()
    return vars

#Define actor-critic network class for PPO algorithm
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate = 3e-4):
        super(ActorCritic, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        self.writer = SummaryWriter()

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
    
#Define the PPO agent class
class PPOAgent:
    def __init__(self, env, client, num_inputs, num_actions, hidden_size, eps_clip, k_epochs = 10, lr = 1e-03, gamma = 0.99):
        self.env = env
        self.client = client
        self.actor_critic = ActorCritic(num_inputs, num_actions, hidden_size, lr)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr = lr)
        self.eps_clip = eps_clip
        self.gammaa = gamma
        self.k_epochs = k_epochs

    def get_action(self, state):
        state = torch.tensor(state, dtype = torch.float32)
        logits, value = self.actor_critic(state)
        action_probs = F.softmax(logits, dim = -1)
        action = action_probs.multinomial(num_samples = 1)
        return action, value
    
    def compute_returns(self, rewards, dones, next_value)
        returns = []
        discounted_reward = next_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        return torch.tensor(returns, dtype = torch.float32)

    #CLIP function
    def clip(self)
    
    def update(self, states, actions, log_probs, returns, advantages):
        for _ in range(self.k_epochs):
            new_log_probs, new_value = self.actor_critic(states)
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = F.mse_loss(new_value, returns)
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            #logging on tensorboard
            self.writer.add_scalar('Loss/actor_loss', actor_loss.mean().item(), self.i_iter)
            self.writer.add_scalar('Loss/critic_loss', critic_loss.mean().item(), self.i_iter)
            self.i_iter += 1

    
