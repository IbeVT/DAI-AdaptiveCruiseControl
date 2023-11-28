import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env

class MyEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = <gym.Space>
        self.observation_space = <gym.Space>
    def reset(self, seed, options):
        return <obs>, <info>
    def step(self, action):
        return <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>

def env_creator(env_config):
    return MyEnv(...)  # return an env instance

register_env("my_env", env_creator)
algo = ppo.PPO(env="my_env")
