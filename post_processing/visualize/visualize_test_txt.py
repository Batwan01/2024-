import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke

# 설정
data_root = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/tld_db/test/images'  # 이미지 경로
prediction_root = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/test_txt/ensemble_codinoobject123_cascade'  # 예측 txt 파일 경로
############

class_names = { 
    0: "veh_go", 1: "veh_goLeft", 2: "veh_noSign", 3: "veh_stop", 4: "veh_stopLeft", 
    5: "vef_stopWarning", 6: "veh_warning", 7: "ped_go", 8: "ped_noSign", 9: "ped_stop", 
    10: "bus_go", 11: "bus_noSign", 12: "bus_stop", 13: "bus_warning"
}  # 클래스 이름

# 클래스별 색상 설정
class_colors = {
    0: (255, 0, 0),    # 빨간색
    1: (0, 255, 0),    # 초록색
    2: (0, 0, 255),    # 파란색
    3: (255, 255, 0),  # 노란색
    4: (255, 0, 255),  # 보라색
    5: (0, 255, 255),  # 청록색
    6: (128, 0, 0),    # 어두운 빨간색
    7: (0, 128, 0),    # 어두운 초록색
    8: (0, 0, 128),    # 어두운 파란색
    9: (128, 128, 0),  # 어두운 노란색
    10: (128, 0, 128), # 어두운 보라색
    11: (0, 128, 128), # 어두운 청록색
    12: (64, 64, 64),  # 회색
    13: (128, 128, 128) # 밝은 회색
}

# YOLO 형식의 txt 파일 읽기 함수
def read_prediction_file(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            elements = line.strip().split()
            class_id = int(elements[0])
            x_center, y_center, width, height, score = map(float, elements[1:])
            predictions.append((class_id, x_center, y_center, width, height, score))
    return predictions

# 텍스트 위치 자동 조정 함수
def find_non_overlapping_position(x1, y1, x2, y2, ax, default_position):
    # 가능한 위치 (상단 좌측, 상단 우측, 하단 좌측, 하단 우측)
    position_options = [
        (x1, y1 - 10),  # 상단 좌측
        (x2, y1 - 10),  # 상단 우측
        (x1, y2 + 20),  # 하단 좌측
        (x2, y2 + 20)   # 하단 우측
    ]

    for pos in position_options:
        # pos가 현재 이미지의 범위를 넘지 않는지 확인하고 텍스트가 겹치지 않는 위치로 선택
        if 0 <= pos[0] < ax.get_xlim()[1] and 0 <= pos[1] < ax.get_ylim()[0]:
            return pos

    return default_position  # 겹치지 않는 위치가 없으면 기본 위치 반환

# 예측 데이터 시각화 함수
def visualize_predictions(image_path, predictions, image_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor='#2f2f2f')
    h, w, _ = cv2.imread(image_path).shape

    for i, (ax, threshold) in enumerate(zip(axes, [0.5, 0.5])):
        # 이미지 읽기 및 설정
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"{image_name} | {'Score >= 0.5' if i == 0 else 'Score < 0.5'}",
                     fontsize=16, color='white', weight='bold')

        # score에 따른 분리된 예측값들
        filtered_predictions = [pred for pred in predictions if (pred[-1] >= threshold) == (i == 0)]
        # 상위 5개 스코어 표시할 예측 선택
        top_predictions = sorted(filtered_predictions, key=lambda x: x[-1], reverse=True)[:5]

        for idx, (class_id, x_center, y_center, width, height, score) in enumerate(filtered_predictions):
            # 바운딩 박스 좌표 변환
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # 클래스별 색상 적용
            color = class_colors.get(class_id, (255, 255, 255))
            color_with_opacity = (color[0] / 255, color[1] / 255, color[2] / 255, 0.6)

            # 바운딩 박스 추가
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color_with_opacity, facecolor='none')
            ax.add_patch(rect)

            # 상위 5개의 바운딩 박스에 대해서만 클래스 번호와 스코어 표시
            if (class_id, x_center, y_center, width, height, score) in top_predictions:
                # 겹치지 않는 위치 찾기
                default_position = (x1, y1 - 10)
                text_position = find_non_overlapping_position(x1, y1, x2, y2, ax, default_position)
                ax.text(
                    text_position[0], text_position[1], f"{class_id}: {score:.2f}",
                    color='white', fontsize=8, weight='bold',  # 텍스트 크기 조정
                    bbox=dict(facecolor=color_with_opacity, alpha=0.6, edgecolor='none', pad=0.1),  # 더 연한 배경 색
                    path_effects=[withStroke(linewidth=1, foreground="black")]  # 테두리 효과 추가
                )

    # 범례 추가 (클래스 번호와 이름 포함)
    patches = [mpatches.Patch(color=[c/255 for c in color], label=f"{class_id}: {class_names[class_id]}") 
               for class_id, color in class_colors.items()]
    plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12, labelcolor='white')

    plt.tight_layout(pad=2)
    plt.show()

# 이미지별 시각화
for prediction_file in os.listdir(prediction_root):
    if prediction_file.endswith('.txt'):
        # 이미지와 예측 파일 경로 설정
        image_name = prediction_file.replace('.txt', '.jpg')  # 타이틀에 사용할 이미지 이름 추출
        image_path = os.path.join(data_root, image_name)
        prediction_path = os.path.join(prediction_root, prediction_file)
        
        # 예측 데이터 읽고 시각화
        predictions = read_prediction_file(prediction_path)
        visualize_predictions(image_path, predictions, image_name)