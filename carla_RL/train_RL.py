import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.tune.search import grid_search
from ray.rllib.algorithms.ppo import PPO
import gym

from setuptools import setup

from gym.envs.registration import register
import gym_carla

import os
print(os.getcwd())
"""register(
    id='carla-v0',
    entry_point='DAI-AdaptiveCruiseControl.rllib-integration2.environment.environment.gym_carla.envs:CarlaEnv',
)"""

"""setup(name='gym_carla',
      version='0.0.1',
      install_requires=['gym', 'pygame']
)"""

params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': True,  # whether to use discrete control space
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

# Set gym-carla environment
env = gym.make('carla-v0', params=params)
env.reset()

if __name__ == "__main__":
    tuner = tune.Tuner(
        PPO,
        tune_config=tune.TuneConfig(max_concurrent_trials=2),
        param_space={
            "framework": "torch",
            # "num_gpus": 0.5,
            "num_workers": 1,
            "env": "CustomCarlaEnv",
            "model":
                {
                    "fcnet_hiddens": [64],
                    "fcnet_activation": "linear",
                },
            "lr": tune.grid_search([5e-3, 5e-4]),
        },
        run_config=train.RunConfig(
            stop={"episode_reward_mean": 30},
            callbacks=[
                WandbLoggerCallback(
                    project="SweepProject",
                    api_key="6370a2f36173950723d7d21b6bad47d74bb7e458",
                )
            ],
        ),
    )
    results = tuner.fit()