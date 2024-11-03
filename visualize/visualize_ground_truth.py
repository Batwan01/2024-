import pandas as pd
import cv2
import os
from tqdm import tqdm

# CSV 파일 경로와 이미지 폴더 경로
csv_file_path = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/val_csv/val_ground_truth.csv'
image_folder_path = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/tld_db/train/images'
output_folder_path = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/visualize_image/ground_truth'

# 클래스 이름과 색상 정의
class_names = { 
    0: "veh_go", 1: "veh_goLeft", 2: "veh_noSign", 3: "veh_stop", 4: "veh_stopLeft", 
    5: "vef_stopWarning", 6: "veh_warning", 7: "ped_go", 8: "ped_noSign", 9: "ped_stop", 
    10: "bus_go", 11: "bus_noSign", 12: "bus_stop", 13: "bus_warning"
}

class_colors = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 0, 0), 7: (0, 128, 0),
    8: (0, 0, 128), 9: (128, 128, 0), 10: (128, 0, 128), 11: (0, 128, 128),
    12: (64, 64, 64), 13: (128, 128, 128)
}

# 출력 폴더가 없으면 생성
os.makedirs(output_folder_path, exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_file_path)
df['PredictionString'] = df['PredictionString'].fillna('')  # NaN을 빈 문자열로 처리

# 모든 클래스에 대해 바운딩 박스를 시각화
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
    image_id = row['image_id']
    prediction_string = row['PredictionString']
    
    # 이미지 경로 설정
    image_path = f"{image_folder_path}/{image_id}"
    image = cv2.imread(image_path)

    if image is None:
        print(f"이미지 {image_id}를 찾을 수 없음")
        continue

    # PredictionString에서 bounding box 정보 추출
    bboxes = prediction_string.split(' ')
    for i in range(0, len(bboxes), 6):
        if i + 5 < len(bboxes):  # 예측 정보가 완전한지 확인
            try:
                class_id = int(bboxes[i])  # 클래스 ID 가져오기
                x1 = float(bboxes[i + 2])
                y1 = float(bboxes[i + 3])
                x2 = float(bboxes[i + 4])
                y2 = float(bboxes[i + 5])

                color = class_colors.get(class_id, (255, 255, 255))
                # 바운딩 박스를 그림
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)
                label = f"{class_names[class_id]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                
                # 텍스트 크기 계산 (텍스트 배경을 그리기 위해 필요)
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_width, text_height = text_size
                
                # 텍스트 배경 사각형 그리기
                background_top_left = (int(x1), int(y1) - text_height - 5)
                background_bottom_right = (int(x1) + text_width, int(y1))
                cv2.rectangle(image, background_top_left, background_bottom_right, color, thickness=-1)
                
                # 텍스트 추가 (하얀색)
                text_position = (int(x1), int(y1) - 5)
                cv2.putText(image, label, text_position, font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

            except ValueError:
                continue  # 변환 실패 시 무시하고 다음으로 진행

    # 이미지에 검은색 테두리 추가 (액자 효과)
    border_thickness = 50
    image_with_border = cv2.copyMakeBorder(
        image, border_thickness, border_thickness, border_thickness, border_thickness, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 상단 중앙에 하얀색 타이틀 추가
    title = f"Image ID: {image_id}"
    title_font_scale = 1
    title_font_thickness = 2
    title_text_size, _ = cv2.getTextSize(title, font, title_font_scale, title_font_thickness)
    title_x = (image_with_border.shape[1] - title_text_size[0]) // 2  # 중앙 정렬
    title_y = border_thickness // 2 + title_text_size[1] // 2         # 테두리 안쪽 중앙

    cv2.putText(
        image_with_border, title, (title_x, title_y), font, title_font_scale, 
        (255, 255, 255), title_font_thickness, lineType=cv2.LINE_AA
    )

    # 시각화된 이미지를 출력 폴더에 저장
    output_path = os.path.join(output_folder_path, f"{image_id}")
    cv2.imwrite(output_path, image_with_border)  # BGR로 저장