import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    '''
    Define custom call back for logging desired values to tensorboard display.
    '''
    def __init__(self, verbose = 0):
        super(TensorboardCallback, self).__init__(verbose)
    