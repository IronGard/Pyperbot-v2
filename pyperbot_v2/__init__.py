#registration of environment
from gymnasium.envs.registration import register 
from .envs.TestEnv import TestEnv
from .envs.ModTestEnv import ModTestEnv
# from .envs.LoaderTestEnv import LoaderTestEnv

register(
    id='SnakebotEnv-v0',
    entry_point='pyperbot_v2.envs:TestEnv'
)

register(
    id = 'ModSnakebotEnv-v0',
    entry_point = 'pyperbot_v2.envs:ModTestEnv'
)

# register(
#     id = 'LoaderSnakebotEnv-v0',
#     entry_point = 'pyperbot_v2.envs:LoaderTestEnv'
# )