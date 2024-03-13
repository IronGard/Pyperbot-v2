import torch
import torch.nn as nn
import torch.optim as optim
from .policy_network import PolicyNetwork

class CustomPPO:
    def __init__(self, env, learning_rate=3e-4, batch_size=64, gamma=0.99, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.policy_network = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def learn(self, total_timesteps, callback=None):
        timestep = 0
        while timestep < total_timesteps:
            batch_obs, batch_actions, batch_rewards, batch_dones, batch_values = self.collect_experiences()
            
            # Calculate advantages
            advantages = self.calculate_advantages(batch_rewards, batch_dones, batch_values)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Perform multiple epochs of PPO optimization
            for _ in range(10):  # Number of epochs
                self.optimize_policy(batch_obs, batch_actions, batch_rewards, batch_dones, batch_values, advantages)
            
            timestep += len(batch_obs)
            
            if callback is not None:
                callback.on_step(timestep)
        
    def collect_experiences(self):
        batch_obs, batch_actions, batch_rewards, batch_dones, batch_values = [], [], [], [], []
        obs = self.env.reset()
        done = False
        while not done:
            action, value = self.policy_network.act(torch.tensor(obs, dtype=torch.float32, device=self.device))
            next_obs, reward, done, _ = self.env.step(action)
            
            batch_obs.append(obs)
            batch_actions.append(action)
            batch_rewards.append(reward)
            batch_dones.append(done)
            batch_values.append(value)
            
            obs = next_obs
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float32, device=self.device)
        batch_values = torch.tensor(batch_values, dtype=torch.float32, device=self.device)
        
        return batch_obs, batch_actions, batch_rewards, batch_dones, batch_values
    
    def calculate_advantages(self, rewards, dones, values):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32, device=self.device)
    
    def optimize_policy(self, obs, actions, rewards, dones, values, advantages):
        # Get the predicted values and action probabilities from the policy network
        predicted_values, action_probs = self.policy_network(obs)
        
        # Calculate the probability ratio
        old_action_probs = action_probs.detach()
        action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        prob_ratio = action_probs / old_action_probs
        
        # Calculate the surrogate loss
        surrogate1 = prob_ratio * advantages
        surrogate2 = torch.clamp(prob_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Calculate the value loss
        value_loss = nn.MSELoss()(predicted_values, rewards)
        
        # Calculate the entropy bonus
        entropy_bonus = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1).mean()
        
        # Calculate the total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_bonus
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()
    
    def save(self, path):
        torch.save(self.policy_network.state_dict(), path)
    
    @classmethod
    def load(cls, path, env):
        agent = cls(env)
        agent.policy_network.load_state_dict(torch.load(path))
        return agent