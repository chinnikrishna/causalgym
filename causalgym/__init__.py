from gym.envs.registration import register
register(
    id='causalgym-task1-v0',
    entry_point='causalgym.envs:Task1Env',
)
