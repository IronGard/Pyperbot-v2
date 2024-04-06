import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor, load_results, ResultsWriter
import numpy as np

class CustomMonitor(gym.wrappers.Monitor):
    def __init__(self, env, filename = None, allow_early_resets = True, reset_keywords = (), info_keywords = ()):
        super(CustomMonitor, self).__init__(env, filename, allow_early_resets, reset_keywords)
        self.custom_info = {}

    def reset(self, **kwargs):
        self.custom_info = {}
        return super(CustomMonitor, self).reset(**kwargs)
    

