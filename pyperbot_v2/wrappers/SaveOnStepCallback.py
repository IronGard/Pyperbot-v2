import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnStepCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.timesteps = []
        self.rewards = []
        self.num_episodes = 0
        self.best_mean_reward = -np.inf

    def _on_step(self):
        # Get the current timestep
        current_timestep = self.model.num_timesteps

        # Record the reward at the current timestep
        self.timesteps.append(current_timestep)
        self.rewards.append(self.training_env.get_attr('episode_rewards')[-1])

        # Check if the episode is complete
        if self.training_env.get_attr('episode_starts')[-1]:
            self.num_episodes += 1

            # Compute the mean reward over the last 100 episodes
            mean_reward = np.mean(self.rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            # Print the progress every 10 episodes
            if self.num_episodes % 10 == 0:
                print(f"Episode: {self.num_episodes}, Best Mean Reward: {self.best_mean_reward:.2f}")

    def _on_training_end(self):
        # Save the recorded timesteps and rewards to a file
        np.savez(os.path.join(self.log_dir, 'rewards.npz'), timesteps=self.timesteps, rewards=self.rewards)

    def plot_rewards(self):
        # Load the recorded timesteps and rewards from the file
        data = np.load(os.path.join(self.log_dir, 'rewards.npz'))
        timesteps = data['timesteps']
        rewards = data['rewards']

        # Plot the rewards vs timesteps
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps, rewards)
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.title('Reward vs Timestep')
        plt.show()