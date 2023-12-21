# DESCRIPTION
This branch tries to combine computer vision of vehicles, signs, traffic lights, etc. to train the RL 
agent (using PPO) to maximise the reward function. 

# OVERVIEW
carla_RL contains everything related to the actual RL. In this folder, Checkpoints contains saved agents, 
environment contains everything related to the interaction with carla, test_RL.py can be used to test 
saved agents using their checkpoint, and train_RL is used to train the PPO agent. The most important 
file is carla_env.py which can be found in carla_RL/environment/environment/gym_carla/envs. This contains 
setting up the carla environment, observations, actions, rewards, etc. All the other files like 
ComputerVision.py and controller2d.py are used in the carla_env.py file to implement the correct 
behaviour.

# HOW TO USE
Steps from when you have a SSH connection:
1) cd ..
2) cd /home/carla/PythonScripts/Stijn/DAI-AdaptiveCruiseControl
3) export DISPLAY=:0
4) git pull
5) pip uninstall gym_carla
6) pip install carla_RL/environment/environment
7) python3 carla_RL/train_RL.py

To test the model:
python3 carla_RL/environment/environment/test.py
(worked two weeks ago, currently crashes CARLA after changes in the carla_env.py file)
