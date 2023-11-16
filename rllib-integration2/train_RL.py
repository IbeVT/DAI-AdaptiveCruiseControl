import ray
from ray import train, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.models import ModelCatalog
from ray.tune.search import grid_search
from ray.rllib.algorithms.dqn import DQN

if __name__ == "__main__":
    tuner = tune.Tuner(
        DQN,
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