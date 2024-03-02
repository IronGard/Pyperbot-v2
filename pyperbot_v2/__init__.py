#registration of environment
from gymnasium.envs.registration import register 
from .envs.TestEnv import TestEnv

register(
    id='SnakebotEnv-v0',
    entry_point='pyperbot_v2.envs:TestEnv'
)

register(
    id = 'ModSnakebotEnv-v0',
    entry_point = 'pyperbot_v2.envs:ModTestEnv'
)