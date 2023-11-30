import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.tune.search import grid_search
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
import gymnasium as gym

from setuptools import setup

from gymnasium.envs.registration import register
from ray.tune.registry import register_env
import sys
sys.path.append('carla_RL/environment/environment')
sys.path.append('carla_RL/environment/environment/gym_carla')
import gym_carla
from gym_carla.envs.carla_env import CarlaEnv
from gymnasium.wrappers import EnvCompatibility


def env_creator(env_config=None):
    print('-----------------------ENV_CREATOR-------------------------\n\n\n')
    return EnvCompatibility(CarlaEnv())

config = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
}

register_env("CustomCarlaEnv", env_creator)

"""register(
    id='CustomCarlaEnv',
    entry_point='gym_carla.envs:CarlaEnv',
)"""

"""setup(name='gym_carla',
      version='0.0.1',
      install_requires=['gym', 'pygame']
)"""



"""# Set gym-carla environment
env = gym.make('CustomCarlaEnv', params=config)
env.reset()"""


if __name__ == "__main__":
    tuner = tune.Tuner(
        PPO,
        tune_config=tune.TuneConfig(max_concurrent_trials=1),
        param_space={
            "framework": "torch",
            # "num_gpus": 0.5,
            "num_workers": 1,
            "env": "CustomCarlaEnv",
            #"env": "CartPole-v1",
            "env_config": {
                "disable_env_checking": True,
            },
            "model":
                {
                    "fcnet_hiddens": [64],
                    "fcnet_activation": "linear",
                },
            "lr": 5e-3   #tune.grid_search([5e-3, 5e-4])
        },
        run_config=train.RunConfig(
            #disable_env_checking=True,
            #stop={"episode_reward_mean": 30},
            callbacks=[
                WandbLoggerCallback(
                    project="SweepProject",
                    api_key="6370a2f36173950723d7d21b6bad47d74bb7e458",
                )
            ],
        ),
    )
    results = tuner.fit()