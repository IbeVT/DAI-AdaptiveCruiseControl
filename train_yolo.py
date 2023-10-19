import comet_ml
from ultralytics import YOLO

comet_ml.init()
# Start from pretrained model
model = YOLO('yolov8n.pt')

# Train the model on the custom dataset
results = model.train(data="./yolo_train.yaml", workers=1)

model.export()

# Validate the model on the test dataset
metrics = model.val()