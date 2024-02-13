from gymnasium.envs.registration import register

register(
    id='CaveEnv-v0',
    entry_point='pyperbot_v2.envs:CaveEnv',
)