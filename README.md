# Adaptive cruise control
This repository contains the code of our CARLA Adaptive Cruise Control project for I-Distributed Artificial Intelligence.

Currently, the most important branches are ComputerVisionTest and RL.

## ComputerVisionTest
This branch contains a working demo of all of the computer vision parts. It can let a car drive in multiple environments/scenario's and shows the images of the camera, overlayed with the radar points and the interpreted results. This includes all bounding boxes found, an indication of which car is the predecessor, the distance to that car, the speed difference with that car and the current speed limit.

## RL
This branch contains the RL environment. The agent can be trained, but it does not learn anything due to problems with the steering controller. While training, it does show the data of the camera, overlayed with the computer vision data.

## Other branches

| Branch | Status | Content | Creator |
| ---- | ---- | ---- | ---- |
| ComputerVision | Deprecated | Attempt to retrieve the sensor info from Carla and make predictions to feed to the RL agent | Ibe |
| Sensors | |Configuring sensors with carla PythonAPI.  | Lukas |
| Yolo | Deprecated |First experiments with and training of the YOLOv8 model | Ibe |
| Dataset-setup | | Gathering data and training the YOLOv8 model | Alberto & María |
| traffic-signs | Deprecated | Training YOLOv8 model on traffic sign recognition & implementation in Carla | Maria |

