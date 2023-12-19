# -*- coding: utf-8 -*-


#Pip install `ultralytics` and [dependencies](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) and check software and hardware.

import comet_ml
import torch
import ultralytics
ultralytics.checks()


from comet_ml import Experiment
experiment = Experiment (api_key="jUuDMMTUiMF5Oz8x6Tza59Iys", project_name="carlaCVFINAL")
comet_ml.init()

from ultralytics import YOLO


# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data='/home/carla/NScripts/alb/DAI-AdaptiveCruiseControl/datasets/3/yolov5_carla.yaml',  name = 'v3CarlaCV', epochs=100, patience=10, batch = 16,
      cache = True, imgsz=640, iou = 0.5,workers=0, save=True,save_period=1,
      augment=False, degrees=0.0, fliplr=0.0, lr0=0.01,weight_decay=0.0005, optimizer='AdamW')  # train the model

results = model.val()  # evaluate model performance on the validation set

#### results = model.test() TEST CODE

#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
#results = model('/content/Carla-Object-Detection-Dataset/Town05_009660.png')  # predict on an image

results = model.export(format='onnx')  # export the model to ONNX format
