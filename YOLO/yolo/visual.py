import pandas as pd
import cv2
import os

# CSV 파일 경로와 이미지 폴더 경로
csv_file_path = '/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolo/runs/detect/result/predictions_yolo11x_val.csv'  # CSV 파일 경로
image_folder_path = '/data/ephemeral/home/dataset/val/images'  # 원본 이미지 폴더 경로
output_folder_path = '/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/YOLO/yolo/runs/detect/result/img'  # 시각화된 이미지를 저장할 폴더 경로

# 출력 폴더가 없으면 생성
os.makedirs(output_folder_path, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_file_path)

# NaN 값을 빈 문자열로 채움
df['PredictionString'] = df['PredictionString'].fillna('')

# 클래스가 13인 레코드 필터링
def has_class_13(prediction_string):
    elements = prediction_string.split(' ')
    for i in range(0, len(elements), 6):
        if i + 5 < len(elements):  # 예측 정보가 완전한지 확인
            try:
                class_id = int(elements[i])  # 클래스 ID 가져오기
                if class_id == 13:  # 클래스 ID가 13인지 확인
                    return True
            except ValueError:
                continue  # 변환 실패 시 무시하고 다음으로 진행
    return False

df_class_13 = df[df['PredictionString'].apply(has_class_13)]

# 클래스 13을 가진 각 이미지를 순회하며 시각화 후 저장
for _, row in df_class_13.iterrows():
    # image_id와 PredictionString 정보를 가져옴
    image_id = row['image_id']
    prediction_string = row['PredictionString']
    
    # PredictionString에서 bounding box 정보 추출
    bboxes = prediction_string.split(' ')
    image_path = f"{image_folder_path}/{image_id}"
    image = cv2.imread(image_path)

    if image is None:
        print(f"이미지 {image_id}를 찾을 수 없음")
        continue

    # BGR을 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i in range(0, len(bboxes), 6):
        if i + 5 < len(bboxes):  # 예측 정보가 완전한지 확인
            try:
                class_id = int(bboxes[i])  # 클래스 ID 가져오기
                score = float(bboxes[i + 1])
                x1 = float(bboxes[i + 2])
                y1 = float(bboxes[i + 3])
                x2 = float(bboxes[i + 4])
                y2 = float(bboxes[i + 5])
                
                # 클래스 ID가 13인 경우에만 바운딩 박스를 그림
                if class_id == 13:
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0), thickness=2)
                    label = f"Class: {class_id}, Score: {score:.2f}"
                    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            except ValueError:
                continue  # 변환 실패 시 무시하고 다음으로 진행

    # 시각화된 이미지를 출력 폴더에 저장
    output_path = os.path.join(output_folder_path, f"visualized_{image_id}")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # 다시 BGR로 변환하여 저장