import os
import pandas as pd
from PIL import Image

txt_dir = "/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/test_txt/predictions"
output_dir = "/Users/imch/workspace/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/test_csv"
csv_output_path = os.path.join(output_dir, "test.csv")
train_data_path = "/Users/imch/workspace/2024-autonomous-driving-artificial-intelligence-challenge/tld_db/train/images"
test_data_path = "/Users/imch/workspace/2024-autonomous-driving-artificial-intelligence-challenge/tld_db/test/images"

mode = 'train' # if train or validation csv = 'train' 
                # if test csv = 'test'

# YOLO 형식의 TXT 파일을 파스칼 형식 CSV 파일로 변환하는 함수 정의
def convert_txt_to_csv(txt_dir, csv_output_path, mode):
    data = []

    # TXT 디렉토리의 모든 파일을 순회
    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            image_name = txt_file.replace('.txt', '.jpg')
            txt_file_path = os.path.join(txt_dir, txt_file)
            
            if mode == 'train':
                image_path = os.path.join(train_data_path, image_name)
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            if mode == 'test':
                image_path = os.path.join(test_data_path, image_name)
                with Image.open(image_path) as img:
                    image_width, image_height = img.size

            prediction_string = []

            # YOLO 형식의 정보를 파스칼 형식으로 변환
            with open(txt_file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * image_width
                    y_center = float(parts[2]) * image_height
                    width = float(parts[3]) * image_width
                    height = float(parts[4]) * image_height
                    score = float(parts[5])
                    
                    # 파스칼 형식으로 변환
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    # 파스칼 형식 문자열 생성
                    prediction_string.extend([
                        f"{class_id}", f"{score:.6f}",
                        f"{x_min:.2f}", f"{y_min:.2f}", f"{x_max:.2f}", f"{y_max:.2f}"
                    ])
            
            # 결과를 리스트에 추가
            data.append({
                "PredictionString": " ".join(prediction_string),
                "image_id": image_name
            })
            print(f"Processed {image_name}")

    # 데이터프레임으로 변환 후 CSV 저장
    df = pd.DataFrame(data)
    df.to_csv(csv_output_path, index=False)
    print(f"CSV 파일로 변환 완료: {csv_output_path}")

# 예제 실행
convert_txt_to_csv(txt_dir, csv_output_path, mode)