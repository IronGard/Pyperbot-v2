#making relevant imports
import numpy as np

import gymnasium as gym
from gymnasium import spaces

#creating a custom environment
class CaveEnv(gym.Env):
    #metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CaveEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64,64,3), dtype=np.uint8)
        self.reward_range = (-1, 1)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
#loading snake robot into the environment