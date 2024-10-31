from ultralytics import YOLO

# Load a model
model = YOLO("ultralytics/cfg/models/v10/yolov10x.yaml")  # build a new model from YAML
model = YOLO("yolov10x.pt")  # load a pretrained model (recommended for training)
model = YOLO("ultralytics/cfg/models/v10/yolov10x.yaml").load("yolov10x.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolo/data.yaml", epochs=20, imgsz=1024, batch=12)