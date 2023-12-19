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
model = YOLO('runs/detect/v2CarlaCV5/weights/last.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(resume=True)

results = model.val()  # evaluate model performance on the validation set

#### results = model.test() TEST CODE

#results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
#results = model('/content/Carla-Object-Detection-Dataset/Town05_009660.png')  # predict on an image

results = model.export(format='onnx')  # export the model to ONNX format
