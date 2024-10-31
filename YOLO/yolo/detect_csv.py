import csv
from ultralytics import YOLO

# Function to write predictions to a CSV file
def write_to_csv(name, pred_str, csv_path):
    data = {
        "PredictionString": pred_str,  # Store the prediction string
        "image_id": name  # Store the image path
    }
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PredictionString", "image_id"])
        if f.tell() == 0:  # If the file is empty, write the header
            writer.writeheader()
        writer.writerow(data)  # Write the prediction data

# 커스텀 YOLO 모델 로드
model = YOLO("/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolo/runs/detect/weight/best_yolo11x.pt")

# 예측을 진행할 이미지 경로
image_folder_path = "/data/ephemeral/home/dataset/test/images"

# 1024 해상도로 예측 (imgsz 인자를 추가)
results = model(image_folder_path, save=True)

# Flag to save results to CSV
save_csv = True
csv_path = "/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolo/runs/detect/result/predictions_yolov11x_test_1280.csv"

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