import csv
from ultralytics import YOLO

# Function to write predictions to a CSV file
def write_to_csv(name, pred_str, csv_path):
    data = {
        "PredictionString": pred_str,  # Store the prediction string
        "image_id": f'test/{name}'  # Store the image path
    }
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PredictionString", "image_id"])
        if f.tell() == 0:  # If the file is empty, write the header
            writer.writeheader()
        writer.writerow(data)  # Write the prediction data

# Load the custom YOLO model
model = YOLO("/data/ephemeral/home/jiwan/level2-objectdetection-cv-18/YOLOv11/runs/detect/train4/weights/best.pt")

# Predict on the test dataset
results = model("/data/ephemeral/home/dataset_origin/test", save=True)

# Flag to save results to CSV
save_csv = True
csv_path = "predictions_1024.csv"

# Process and save predictions
for result in results:
    pred_str = ""  # Initialize prediction string
    for box in result.boxes:
        cls = int(box.cls.item())  # Class ID
        conf = box.conf.item()  # Confidence score
        xyxy = [coord.item() for coord in box.xyxy[0]]  # Bounding box coordinates
        pred_str += f"{cls} {conf} {xyxy[0]} {xyxy[1]} {xyxy[2]} {xyxy[3]} "  # Add predictions

    if save_csv:
        write_to_csv(result.path.split('/')[-1], pred_str.strip(), csv_path)

print(f"Predictions saved to {csv_path}")
