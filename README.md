# Adaptive cruise control
This repository contains the code of our CARLA Adaptive Cruise Control project for I-Distributed Artificial Intelligence.

Currently, there are only parts of the code, seperated in multiple branches as follows:

| Branch | Content | Creator |
| ---- | ---- | ---- |
| RL | Attempt to create a RL agent | Stijn |
| ComputerVision | Attempt to retrieve the sensor info from Carla and make predictions to feed to the RL agent | Ibe |
| SensorLukas | Learning how to use the CARLA API | Lukas |
| SensorAlexander | Learning how to use the CARLA API | Alexander |
| Yolo | First experiments with and training of the YOLOv8 model | Ibe |
| Dataset-setup | Gathering data and training the YOLOv8 model | Alberto & María |


#DatasetSetup

DataSetCreation/setVar.sh  - Script to set all the enviroment variables needed for our dataset creation

DataSetCreation/carla__tool/carla_dataset_tools/fullDataset.sh - Script to run multiple scenarios for 1 hour each, recording from 6 different cars

DataSetCreation/carla__tool/carla_dataset_tools/label_tools/yolo_label.py - labelling tool that creates our dataset (-r recordFolder)

DataSetCreation/UniqueNamer.sh.sh -Script to create unique names for each image in the dataset

DataSetCreation/NameFix.sh  - Script to make files easier to read for model training

modelTraining.py - python script to train our model
