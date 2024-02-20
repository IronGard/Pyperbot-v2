#Defines utility functions for the test environment
import numpy as np

def reset_snake_robot(env):
    '''
    Function to reset the snake robot in the environment
    '''
    env.reset()

def random_action(env):
    '''
    Function to take random action in the environment
    '''
    return env.action_space.sample()

def rollout(env, agent, max_steps = 1000):
    '''
    Function to roll out the environment with the agent
    '''
    obs = env.reset()
    total_reward = 0
    for i in range(max_steps):
        action, _ = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def evaluate_agent(env, agent, n_eval_episodes = 10):
    '''
    Evaluate agent's performance by simulating multiple rollouts
    '''
    total_rewards = []
    for _ in range(n_eval_episodes):
        total_reward = rollout(env, agent)
        total_rewards.append(total_reward)
    mean_reward = np.mean(total_rewards)
    return mean_reward