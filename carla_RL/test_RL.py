import ray
from ray import train, tune, air
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.tune.search import grid_search
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

import carla
import ray.rllib.algorithms.ppo as ppo

"""def env_creator(env_config=None):
    print('-----------------------ENV_CREATOR-------------------------\n\n\n')
    return EnvCompatibility(CarlaEnv())

register_env("CustomCarlaEnv", env_creator)"""

def main():
  # load and restore model
  agent = ppo.PPO(env="CustomCarlaEnv")
  agent.restore('/home/carla/PythonScripts/Stijn/DAI-AdaptiveCruiseControl/carla_RL/Checkpoints')
  print(f"Agent loaded from saved model at {'/home/carla/PythonScripts/Stijn/DAI-AdaptiveCruiseControl/carla_RL/Checkpoints'}")

  # inference
  env = gym.make("CustomCarlaEnv")
  obs, info = env.reset()
  while True:
      action = agent.compute_single_action(obs)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
          print(f"Agent Completed Episode -> TERMINATION & RESET.")
          env.reset()


if __name__ == '__main__':
  main()