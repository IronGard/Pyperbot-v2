import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnStepCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.timesteps = []
        self.episode_rewards = []
        self.accumulated_reward = 0
        self.episode_lengths = []
        self.episode_timesteps = []
        self.num_episodes = 0
        self.best_mean_reward = -np.inf

    def _on_step(self):
        # Get the current timestep
        current_timestep = self.model.num_timesteps

        # Accumulate the reward for the current timestep
        self.accumulated_reward += self.training_env.get_attr('reward')[0]

        # Check if the episode is done
        done = self.training_env.get_attr('done')[0]
        if done:
            # Append the accumulated reward to the episode rewards list
            self.episode_rewards.append(self.accumulated_reward)
            self.accumulated_reward = 0

            # Append the episode length and timestep to the respective lists
            self.episode_lengths.append(self.training_env.get_attr('episode_length')[0])
            self.episode_timesteps.append(current_timestep)

            self.num_episodes += 1

            # Compute the mean reward over the last 100 episodes
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            # Print the progress every 10 episodes
            if self.num_episodes % 10 == 0:
                print(f"Episode: {self.num_episodes}, Best Mean Reward: {self.best_mean_reward:.2f}")

    def _on_training_end(self):
        # Save the episode rewards, lengths, and timesteps to files
        np.savez(os.path.join(self.log_dir, 'rewards.npz'), episode_rewards=self.episode_rewards,
                 episode_lengths=self.episode_lengths, episode_timesteps=self.episode_timesteps)

    def plot_rewards(self):
        # Load the episode rewards, lengths, and timesteps from the file
        data = np.load(os.path.join(self.log_dir, 'rewards.npz'))
        episode_rewards = data['episode_rewards']
        episode_lengths = data['episode_lengths']
        episode_timesteps = data['episode_timesteps']

        # Plot the rewards vs timesteps
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(episode_timesteps, episode_rewards)
        plt.xlabel('Timestep')
        plt.ylabel('Episode Reward')
        plt.title('Episode Reward vs Timestep')
        plt.show()