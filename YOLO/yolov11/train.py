from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.yaml")  # build a new model from YAML
model = YOLO("yolo11x.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11x.yaml").load("yolo11x.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolov11/data.yaml", epochs=25, imgsz=1024, batch=8)