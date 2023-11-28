from gym.envs.registration import register

register(
    id='CustomCarlaEnv',
    entry_point='gym_carla.envs:CarlaEnv',
)