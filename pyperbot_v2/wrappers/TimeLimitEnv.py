import gymnasium as gym

class TimeLimitEnv(gym.Wrapper):
    '''
    Enforces a step limit on an environment, where episode terminates after
    '''
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimitEnv, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def render(self, mode='human', **kwargs):
        return self.env.render(mode, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)