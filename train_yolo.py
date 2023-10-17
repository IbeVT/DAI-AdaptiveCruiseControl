from ultralytics import YOLO

# Start from pretrained model
model = YOLO('yolov8n.pt')

# Train the model on the custom dataset
results = model.train(data="./yolo_train.yaml", workers=0)
print("results:\n" + results)

model.export()

# Validate the model on the test dataset
metrics = model.val()
print("Validation metric\n" + metrics)
