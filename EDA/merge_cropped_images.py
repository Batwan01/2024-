import cv2
import os
import numpy as np
from tqdm import tqdm

def merge_cropped_images(input_dir, output_dir, images_per_merge=4):
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')
    
    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(input_image_dir))
    
    for i in tqdm(range(0, len(image_files), images_per_merge)):
        batch = image_files[i:i+images_per_merge]
        if len(batch) < images_per_merge:
            break  # 마지막 배치가 4개 미만이면 처리하지 않음
        
        images = []
        labels = []
        heights = []
        widths = []
        
        for img_file in batch:
            img_path = os.path.join(input_image_dir, img_file)
            label_path = os.path.join(input_label_dir, os.path.splitext(img_file)[0] + '.txt')
            
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            images.append(img)
            heights.append(height)
            widths.append(width)
            
            with open(label_path, 'r') as f:
                labels.append(f.read().strip())
        
        max_height = max(heights)
        max_width = max(widths)
        
        # 2x2 그리드 생성
        grid = np.zeros((max_height*2, max_width*2, 3), dtype=np.uint8)
        new_labels = []
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            row = idx // 2
            col = idx % 2
            h, w = img.shape[:2]
            grid[row*max_height:(row*max_height)+h, col*max_width:(col*max_width)+w] = img
            
            # 레이블 조정
            class_id, x_center, y_center, width, height = map(float, label.split())
            
            # 원본 이미지에서의 실제 픽셀 좌표 계산
            x_pixel = x_center * w
            y_pixel = y_center * h
            width_pixel = width * w
            height_pixel = height * h
            
            # 새 그리드에서의 좌표 계산
            new_x_center = (col * max_width + x_pixel) / (max_width * 2)
            new_y_center = (row * max_height + y_pixel) / (max_height * 2)
            new_width = width_pixel / (max_width * 2)
            new_height = height_pixel / (max_height * 2)
            
            new_labels.append(f"{int(class_id)} {new_x_center} {new_y_center} {new_width} {new_height}")
        
        # 새 이미지 저장
        output_image_path = os.path.join(output_image_dir, f"merged_{i//images_per_merge}.jpg")
        cv2.imwrite(output_image_path, grid)
        
        # 새 레이블 저장
        output_label_path = os.path.join(output_label_dir, f"merged_{i//images_per_merge}.txt")
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(new_labels))

# 사용 예
input_dir = '../ultralytics/datasets/TLD_2024/tld_db/train_cropped'
output_dir = '../ultralytics/datasets/TLD_2024/tld_db/train_merged'
merge_cropped_images(input_dir, output_dir)