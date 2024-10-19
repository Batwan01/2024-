import cv2
import os
import numpy as np
from tqdm import tqdm

def find_max_box_size(label_dir):
    max_size = 0
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    _, _, _, width, height = map(float, line.strip().split())
                    max_size = max(max_size, width, height)
    return max_size

def crop_and_save_boxes(image_dir, label_dir, output_dir, max_box_size, padding_factor=0.2, max_crop_size=300):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    image_files = sorted(os.listdir(image_dir))
    for image_file in tqdm(image_files):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_number = int(os.path.splitext(image_file)[0])
            if image_number > 21533:
                break

            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
            
            if not os.path.exists(label_path):
                continue

            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                box_width = int(width * img_width)
                box_height = int(height * img_height)
                
                box_size = max(box_width, box_height)
                padding = int(box_size * padding_factor)
                crop_size = min(box_size + 2 * padding, max_crop_size)
                half_size = crop_size // 2

                center_x = int(x_center * img_width)
                center_y = int(y_center * img_height)

                x1 = max(0, center_x - half_size)
                y1 = max(0, center_y - half_size)
                x2 = min(img_width, center_x + half_size)
                y2 = min(img_height, center_y + half_size)

                cropped_img = image[y1:y2, x1:x2]

                new_width = width * img_width / (x2 - x1)
                new_height = height * img_height / (y2 - y1)
                new_x_center = (center_x - x1) / (x2 - x1)
                new_y_center = (center_y - y1) / (y2 - y1)

                output_image_path = os.path.join(output_dir, 'images', f"{os.path.splitext(image_file)[0]}_{i}.jpg")
                cv2.imwrite(output_image_path, cropped_img)

                output_label_path = os.path.join(output_dir, 'labels', f"{os.path.splitext(image_file)[0]}_{i}.txt")
                with open(output_label_path, 'w') as f:
                    f.write(f"{int(class_id)} {new_x_center} {new_y_center} {new_width} {new_height}\n")

# 사용 예
dataset_dir = '../ultralytics/datasets/TLD_2024/tld_db/train'
image_dir = os.path.join(dataset_dir, 'images')
label_dir = os.path.join(dataset_dir, 'labels')
output_dir = '../ultralytics/datasets/TLD_2024/tld_db/train_cropped'

max_box_size = find_max_box_size(label_dir)
crop_and_save_boxes(image_dir, label_dir, output_dir, max_box_size, padding_factor=2.0, max_crop_size=300)