import os
import json
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

def convert_bbox_yolo2coco(img_width, img_height, bbox):
    """
    YOLO 형식의 바운딩 박스를 COCO 형식으로 변환합니다.
    YOLO bbox: [x_center, y_center, width, height] (모두 0~1로 정규화)
    COCO bbox: [x_min, y_min, width, height]
    """
    x_center, y_center, width, height = bbox
    
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    
    return [x_min, y_min, width, height]

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

def yolo2coco(input_path, classes_file, output_json, image_dir):
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    images = []
    annotations = []
    
    image_id = 0
    annotation_id = 0
    
    for filename in tqdm(os.listdir(input_path), desc="Converting annotations"):
        if filename.endswith('.txt'):
            image_filename = os.path.splitext(filename)[0] + '.jpg'
            image_path = os.path.join(image_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            
            width, height = get_image_size(image_path)
            
            with open(os.path.join(input_path, filename), 'r') as f:
                lines = f.readlines()
            
            images.append({
                "id": image_id,
                "file_name": f"train/images/{image_filename}",  # 'train/images/' 추가
                "width": width,
                "height": height
            })
            
            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())
                bbox = convert_bbox_yolo2coco(width, height, [x_center, y_center, w, h])
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                
                annotation_id += 1
            
            image_id += 1
    
    categories = [{"id": i, "name": name} for i, name in enumerate(classes)]
    
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

def split_train_val(coco_data, split_image):
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    
    split_image_full = f"train/images/{split_image}"
    
    for img in coco_data['images']:
        if img['file_name'] <= split_image_full:
            train_data['images'].append(img)
        else:
            val_data['images'].append(img)
    
    train_image_ids = set(img['id'] for img in train_data['images'])
    
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_image_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)
    
    train_data['categories'] = coco_data['categories']
    val_data['categories'] = coco_data['categories']
    
    return dict(train_data), dict(val_data)


def create_test_coco(test_image_dir, output_json):
    images = []
    image_id = 0
    
    for filename in tqdm(os.listdir(test_image_dir), desc="Processing test images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_image_dir, filename)
            width, height = get_image_size(image_path)
            
            images.append({
                "id": image_id,
                "file_name": f"test/images/{filename}",  # 'test/images/' 추가
                "width": width,
                "height": height
            })
            
            image_id += 1
    
    coco_format = {
        "images": images,
        "annotations": [],
        "categories": []  # 테스트 데이터에는 카테고리 정보가 필요 없을 수 있습니다.
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

# 실행
input_path = '/hdd1/lim_data/tld_db/train/labels'
classes_file = '/hdd1/lim_data/tld_db/train/classes.txt'
output_json = '/hdd1/lim_data/tld_db/json/train_coco.json'
image_dir = '/hdd1/lim_data/tld_db/train/images'

yolo2coco(input_path, classes_file, output_json, image_dir)

# train_coco.json 읽기
with open(output_json, 'r') as f:
    coco_data = json.load(f)

# train과 val 분리
train_data, val_data = split_train_val(coco_data, '00021533.jpg')

# 데이터 확인
print(f"Train images: {len(train_data['images'])}")
print(f"Train annotations: {len(train_data['annotations'])}")
print(f"Val images: {len(val_data['images'])}")
print(f"Val annotations: {len(val_data['annotations'])}")


# train_coco.json과 val_coco.json 저장
with open('../../tld_db/json/train_coco.json', 'w') as f:
    json.dump(train_data, f)
with open('../../tld_db/json/val_coco.json', 'w') as f:
    json.dump(val_data, f)

# test 데이터 처리
test_output_json = '../../tld_db/json/test_coco.json'
test_image_dir = '../../tld_db/test/images'

create_test_coco(test_image_dir, test_output_json)

print("All data processed and saved.")