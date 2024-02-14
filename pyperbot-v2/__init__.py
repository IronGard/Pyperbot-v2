from gymnasium.envs.registration import register

register(
    id='SnakebotEnv-v0',
    entry_point='pyperbot_v2.envs:SnakebotEnv',
)