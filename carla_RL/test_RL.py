#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gymnasium as gym
import gym_carla
import carla
from ray import tune, air
import ray.rllib.algorithms.ppo as ppo

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