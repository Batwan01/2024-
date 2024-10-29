# csv (파스칼) -> txt (YOLO)
'''
csv (파스칼) form
    클래스번호1 스코어1 x_min y_min x_max y_max 클래스번호2 스코어2 x_min y_min x_max y_max | image_name_1
    클래스번호3 스코어3 x_min y_min x_max y_max 클래스번호3 스코어3 x_min y_min x_max y_max | image_name_2
    ...
'''

'''
txt (YOLO) form
prediction
    image_name_1.txt
        클래스번호1 x_center y_center width height 스코어1
        클래스번호2 x_center y_center width height 스코어2

    image_name_2.txt
        클래스번호3 x_center y_center width height 스코어3
        클래스번호4 x_center y_center width height 스코어5
'''

import os
import pandas as pd
from PIL import Image

csv_file_path = '/Users/imch/Downloads/Co-DETR(SwinL, lsj, 3ep)_val.csv'
output_dir = "/Users/imch/workspace/2024-autonomous-driving-artificial-intelligence-challenge/yolo_txt_file/predictions"
train_data_path = "/Users/imch/workspace/2024-autonomous-driving-artificial-intelligence-challenge/tld_db/train/images"
test_data_path = "/Users/imch/workspace/2024-autonomous-driving-artificial-intelligence-challenge/tld_db/test/images"

mode = 'train' # if train or validation csv = 'train' 
                # if test csv = 'test'

# CSV 파일(파스칼 형식)을 TXT(YOLO 형식)으로 변환하는 함수 정의
def convert_csv_to_txt(csv_file_path, output_dir, mode):
    # CSV 파일 불러오기
    csv_data = pd.read_csv(csv_file_path)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 행마다 YOLO 형식으로 변환
    for _, row in csv_data.iterrows():

        if mode == 'train':
            image_path = os.path.join(train_data_path, row['image_id'])
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        if mode == 'test':
            image_path = os.path.join(test_data_path, row['image_id'])
            with Image.open(image_path) as img:
                image_width, image_height = img.size

        # 이미지 이름과 객체 정보 분리
        image_name = row['image_id'].split('/')[-1].replace('.jpg', '')
        objects_info = row['PredictionString'].split()
        
        yolo_lines = []
        
        # 객체 정보 반복 처리
        for i in range(0, len(objects_info), 6):
            class_id = int(objects_info[i])
            score = float(objects_info[i + 1])
            x_min = float(objects_info[i + 2])
            y_min = float(objects_info[i + 3])
            x_max = float(objects_info[i + 4])
            y_max = float(objects_info[i + 5])
            
            # YOLO 형식의 x_center, y_center, width, height 계산
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min)  / image_width
            height = (y_max - y_min) / image_height
            
            # YOLO 형식의 객체 정보 추가
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}")
        
        # YOLO 형식 텍스트 파일 생성 및 저장
        with open(os.path.join(output_dir, f"{image_name}.txt"), "w") as txt_file:
            txt_file.write("\n".join(yolo_lines))
        print(f"complete_{image_name}")

# 예제 실행
convert_csv_to_txt(csv_file_path, output_dir, mode)