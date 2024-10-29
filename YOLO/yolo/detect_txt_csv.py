import os
from ultralytics import YOLO

# Function to write predictions to a txt file in YOLO format
def write_to_txt(name, boxes, txt_dir):
    txt_path = os.path.join(txt_dir, f"{name}.txt")
    
    with open(txt_path, mode="w") as f:
        for box in boxes:
            cls = int(box.cls.item())  # Class ID
            #conf = box.conf.item()  # Confidence score
            x_center = (box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2
            y_center = (box.xyxy[0][1].item() + box.xyxy[0][3].item()) / 2
            width = box.xyxy[0][2].item() - box.xyxy[0][0].item()
            height = box.xyxy[0][3].item() - box.xyxy[0][1].item()
            
            # Convert to relative coordinates (0~1)
            img_width = result.orig_shape[1]
            img_height = result.orig_shape[0]
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height
            
            # Write in YOLO format: class x_center y_center width height
            f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

# Load the custom YOLO model
model = YOLO("/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolov11/runs/detect/train3/weights/best.pt")

# Predict on the test dataset
results = model("/data/ephemeral/home/dataset/test/images", save=True)

# Directory to save the txt files
txt_dir = "yolo_predictions"
os.makedirs(txt_dir, exist_ok=True)

# Process and save predictions in YOLO format
for result in results:
    boxes = result.boxes
    image_name = result.path.split('/')[-1].split('.')[0]  # Get image name without extension
    write_to_txt(image_name, boxes, txt_dir)

print(f"Predictions saved in YOLO format in the directory: {txt_dir}")