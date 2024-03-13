import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

#simple implementation of policy network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, obs):
        action_probs = self.actor(obs)
        value = self.critic(obs)
        return value, action_probs
    
    def act(self, obs):
        with torch.no_grad():
            value, action_probs = self.forward(obs)
            dist = Categorical(action_probs)
            action = dist.sample()
        return action.item(), value.item()