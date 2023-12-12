import gymnasium as gym
import gym_carla
import carla


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
import wandb

def env_creator(env_config=None):
    print('-----------------------ENV_CREATOR-------------------------\n\n\n')
    return EnvCompatibility(CarlaEnv())

register_env("CustomCarlaEnv", env_creator)

if __name__ == "__main__":
    #ray.init(local_mode=True)

    tuner = tune.Tuner(
        PPO,
        tune_config=tune.TuneConfig(
            max_concurrent_trials=1,
        ),
        param_space={
            "disable_env_checking": True,
            "ignore_workers_failure": False,
            "max_concurrent_trials": 1,
            "framework": "torch",
            "num_workers": 0,
            "env": "CustomCarlaEnv",
            #"env": "CartPole-v1",
            "env_config": {
                "disable_env_checking": True,
                "ignore_workers_failure": False,
            },
            "model":
                {
                    "fcnet_hiddens": [1],
                    "fcnet_activation": "linear",
                },
            "lr": 5e-4   #tune.grid_search([5e-3, 5e-4])
        },
        run_config=train.RunConfig(
            stop={"episode_reward_mean": 500},
            local_dir='/home/carla/PythonScripts/Stijn/DAI-AdaptiveCruiseControl/carla_RL/Checkpoints',
            checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True, checkpoint_frequency=1),
            callbacks=[
                WandbLoggerCallback(
                    project="CarlaRL",
                    api_key="cee1795c4e0d51b4eb7fa2b4f7f180c85403aae1",
                )
            ],
        ),
    )
    results = tuner.fit()

    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Trained model saved at {best_result}")

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    print('Best checkpoint:', best_checkpoint)

    # load and restore model
    agent = ppo.PPO(env=env_name)
    agent.restore(checkpoint_path)
    print(f"Agent loaded from saved model at {checkpoint_path}")


def main():
  # parameters for the gym_carla environment
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
  env = gym.make('CustomCarlaEnv')
  env.reset()

  while True:
    action = [2.0, 0.0]
    obs,r,done,info = env.step(action)

    if done:
      env.reset()


if __name__ == '__main__':
  main()