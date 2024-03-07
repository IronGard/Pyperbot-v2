#Environment for normalised actions
import gymnasium as gym

class NormaliseEnv(gym.Wrapper):
    def __init__(self, env):
        super(NormaliseEnv, self).__init__(env)
        self.action_space = gym.spaces.Box(low = -1, high = 1, shape = env.action_space.shape, dtype = env.action_space.dtype)
    
    def step(self, action):
        #normalise the action
        action = action * (self.action_space.high - self.action_space.low) / 2.0 + (self.action_space.high + self.action_space.low) / 2.0
        return self.env.step(action)
    
    def reset(self):
        return self.env.reset()