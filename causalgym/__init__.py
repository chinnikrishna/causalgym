from gym.envs.registration import register
register(
    id='task1-v0',
    entry_point='causalgym.envs:Task1Env',
)
