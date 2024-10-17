import json
import os
from tqdm import tqdm
import yaml
import random
import shutil
import numpy as np

def convert_bbox_coco2yolo(img_width, img_height, bbox):
    """
    COCO 형식의 바운딩 박스를 YOLO 형식으로 변환합니다.
    COCO bbox: [x_min, y_min, width, height]
    YOLO bbox: [x_center, y_center, width, height] (모두 0~1로 정규화)
    """
    x_min, y_min, width, height = bbox

    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width = width / img_width
    height = height / img_height

    return [x_center, y_center, width, height]

def split_coco_data(json_file, train_ratio=0.8, random_seed=42):
    with open(json_file, 'r') as f:
        data = json.load(f)

    random.seed(random_seed)
    
    # 이미지 ID 목록을 섞습니다
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)

    # 8:2 비율로 분할
    split_index = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_index])
    val_ids = set(image_ids[split_index:])

    # 분할된 데이터 생성
    train_data = {
        'images': [img for img in data['images'] if img['id'] in train_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_ids],
        'categories': data['categories']
    }
    val_data = {
        'images': [img for img in data['images'] if img['id'] in val_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_ids],
        'categories': data['categories']
    }

    return train_data, val_data

def coco2yolo(coco_data, output_path):
    """
    COCO 데이터를 YOLO 형식으로 변환합니다.
    """
    os.makedirs(output_path, exist_ok=True)

    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    category_id_to_index = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    for img in tqdm(coco_data['images'], desc="Converting annotations"):
        img_id = img['id']
        img_name = image_id_to_name[img_id]
        img_width = img['width']
        img_height = img['height']

        label_name = os.path.splitext(os.path.basename(img_name))[0] + '.txt'
        label_path = os.path.join(output_path, label_name)

        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

        with open(label_path, 'w') as f:
            for ann in annotations:
                category_index = category_id_to_index[ann['category_id']]
                bbox = convert_bbox_coco2yolo(img_width, img_height, ann['bbox'])
                f.write(f"{category_index} {' '.join([str(x) for x in bbox])}\n")

    print(f"Conversion complete. YOLO format labels saved in {output_path}")
    return coco_data['categories']

def copy_images(coco_data, src_dir, dst_dir):
    """
    이미지를 소스 디렉토리에서 대상 디렉토리로 복사합니다.
    """
    os.makedirs(dst_dir, exist_ok=True)
    for img in tqdm(coco_data['images'], desc=f"Copying images to {dst_dir}"):
        file_name = img['file_name']
        if file_name.startswith('train/') or file_name.startswith('test/'):
            file_name = os.path.basename(file_name)
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Source file not found: {src_path}")

# 실행
train_data, val_data = split_coco_data('../dataset/json/train.json', train_ratio=0.8, random_seed=42)

# train_yolo.json과 val_yolo.json 저장
with open('../dataset/json/train_yolo.json', 'w') as f:
    json.dump(train_data, f)
with open('../dataset/json/val_yolo.json', 'w') as f:
    json.dump(val_data, f)

# 이미지 복사
copy_images(train_data, '../dataset/train', '../dataset/images/train_yolo')
copy_images(val_data, '../dataset/train', '../dataset/images/val_yolo')

# COCO 형식을 YOLO 형식으로 변환
train_categories = coco2yolo(train_data, '../dataset/labels/train_yolo')
val_categories = coco2yolo(val_data, '../dataset/labels/val_yolo')

# YOLO 데이터셋 설정 파일 생성
dataset_config = {
    'path': '../../dataset',
    'train': 'images/train_yolo',
    'val': 'images/val_yolo',
    'test': 'images/test',
    'nc': len(train_categories),  # 클래스 수
    'names': [cat['name'] for cat in train_categories]
}

with open('../dataset/yaml/dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

print("Dataset configuration saved as dataset.yaml")

# test 데이터 처리
with open('../dataset/json/test.json', 'r') as f:
    test_data = json.load(f)
test_categories = coco2yolo(test_data, '../dataset/labels/test')
copy_images(test_data, '../dataset/test', '../dataset/images/test')

print("All data processed and saved.")