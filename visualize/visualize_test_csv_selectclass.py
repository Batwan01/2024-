import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
import pandas as pd
import gc
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')

# 설정
data_root = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/tld_db/test/images'  # 이미지 경로
csv_path = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/test_csv/nmw0.4(Co-DETR(obj365, 3ep), Cascade(5ep, 2ep_over), Co-DETR(obj, 2048, 1ep, over)_test_0filtered.csv'  # CSV 파일 경로
save_dir = '/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/visualize_image/test_leaderboard_4_bus'
class_names = { 
    0: "veh_go", 1: "veh_goLeft", 2: "veh_noSign", 3: "veh_stop", 4: "veh_stopLeft", 
    5: "vef_stopWarning", 6: "veh_warning", 7: "ped_go", 8: "ped_noSign", 9: "ped_stop", 
    10: "bus_go", 11: "bus_noSign", 12: "bus_stop", 13: "bus_warning"
}  # 클래스 이름

class_filter = [13]  

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

# CSV 파일 로드 및 파싱 함수
def parse_csv_data(csv_path):
    df = pd.read_csv(csv_path, header=None)
    df = df.sort_values(by=1).reset_index(drop=True)
    data = []

    for _, row in df.iterrows():
        if row[0] == 'PredictionString':  # 헤더 무시
            continue
        image_name = row[1]
        predictions = []
        items = row[0].split()
        for i in range(0, len(items), 6):
            class_id = int(items[i])
            score = float(items[i + 1])
            x_min = float(items[i + 2])
            y_min = float(items[i + 3])
            x_max = float(items[i + 4])
            y_max = float(items[i + 5])
            predictions.append((class_id, score, x_min, y_min, x_max, y_max))
        data.append((image_name, predictions))
    return data

# 텍스트 위치 자동 조정 함수
def find_non_overlapping_position(x1, y1, x2, y2, ax, default_position):
    position_options = [
        (x1, y1 - 10),  # 상단 좌측
        (x2, y1 - 10),  # 상단 우측
        (x1, y2 + 20),  # 하단 좌측
        (x2, y2 + 20)   # 하단 우측
    ]
    for pos in position_options:
        if 0 <= pos[0] < ax.get_xlim()[1] and 0 <= pos[1] < ax.get_ylim()[0]:
            return pos
    return default_position

# 예측 데이터 시각화 함수
def visualize_predictions(image_path, predictions, image_name, folder_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor='#2f2f2f')
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, (ax, threshold) in enumerate(zip(axes, [0.5, 0.5])):

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"{image_name} | {'Score >= 0.5' if i == 0 else 'Score < 0.5'}",
                     fontsize=16, color='white', weight='bold')

        # score에 따른 분리된 예측값들
        filtered_predictions = [pred for pred in predictions if (pred[1] >= threshold) == (i == 0)]
        top_predictions = sorted(filtered_predictions, key=lambda x: x[1], reverse=True)[:5]

        for idx, (class_id, score, x_min, y_min, x_max, y_max) in enumerate(filtered_predictions):
            color = class_colors.get(class_id, (255, 255, 255))
            color_with_opacity = (color[0] / 255, color[1] / 255, color[2] / 255, 0.6)
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color_with_opacity, facecolor='none')
            ax.add_patch(rect)

            if (class_id, score, x_min, y_min, x_max, y_max) in top_predictions:
                default_position = (x_min, y_min - 10)
                text_position = find_non_overlapping_position(x_min, y_min, x_max, y_max, ax, default_position)
                ax.text(
                    text_position[0], text_position[1], f"{class_id}: {score:.2f}",
                    color='white', fontsize=8, weight='bold',
                    bbox=dict(facecolor=color_with_opacity, alpha=0.4, edgecolor='none', pad=0.1),
                    path_effects=[withStroke(linewidth=1, foreground="black")]
                )

    patches = [mpatches.Patch(color=[c/255 for c in color], label=f"{class_id}: {class_names[class_id]}")
               for class_id, color in class_colors.items()]
    plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=12, labelcolor='white')
    plt.tight_layout(pad=2)

    # **추가된 코드: 이미지 저장 경로 생성 및 저장**
    if not os.path.exists(os.path.join(save_dir, folder_name)):
        os.makedirs(os.path.join(save_dir, folder_name))  # 저장 폴더가 없으면 생성
    save_path = os.path.join(save_dir, folder_name, f"{image_name}")  # 저장 경로 설정
    plt.savefig(save_path, bbox_inches='tight', facecolor=fig.get_facecolor())  # 저장
    plt.close(fig)

# 시각화 실행
data = parse_csv_data(csv_path)
for i, (image_name, predictions) in enumerate(tqdm(data, desc="Processing Images")):
    image_path = os.path.join(data_root, image_name)
    if os.path.exists(image_path):
        # 특정 클래스가 포함된 이미지인지 확인
        if any(class_id in class_filter for class_id, *_ in predictions):
            # 선택한 클래스에 맞는 폴더 생성 및 시각화
            for class_id in class_filter:
                if any(pred[0] == class_id for pred in predictions):
                    folder_name = os.path.join(save_dir, class_names[class_id])
                    visualize_predictions(image_path, predictions, image_name, folder_name)
    if i % 1000 == 0:
        gc.collect()