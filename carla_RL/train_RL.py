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
            # "num_gpus": 0.5,
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
            "lr": 0.0001   #tune.grid_search([5e-3, 5e-4])
        },
        run_config=train.RunConfig(
            #ignore_workers_failures=False,
            #disable_env_checking=True,
            #stop={"episode_reward_mean": 500},
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
    agent = PPO(env='CustomCarlaEnv')
    agent.restore(checkpoint_path)
    print(f"Agent loaded from saved model at {checkpoint_path}")

