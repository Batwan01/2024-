import os
import shutil

# 디렉토리 경로 설정
train_images_dir = "/data/ephemeral/home/dataset/train/images"
train_labels_dir = "/data/ephemeral/home/dataset/train/labels"
val_images_dir = "/data/ephemeral/home/dataset/val/images"
val_labels_dir = "/data/ephemeral/home/dataset/val/labels"

# 이동할 파일 번호 범위 설정
start_num = 21534
end_num = 26863

# val/images 및 val/labels 폴더가 없을 경우 생성
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 이미지 파일 이동
for num in range(start_num, end_num + 1):
    image_filename = f"{num:08d}.jpg"
    image_src = os.path.join(train_images_dir, image_filename)
    image_dst = os.path.join(val_images_dir, image_filename)
    # 파일이 존재하면 이동
    if os.path.exists(image_src):
        shutil.move(image_src, image_dst)
        print(f"Moved {image_src} to {image_dst}")
    else:
        print(f"Image file {image_src} not found.")
        
# 라벨 파일 이동
for num in range(start_num, end_num + 1):
    label_filename = f"{num:08d}.txt"
    label_src = os.path.join(train_labels_dir, label_filename)
    label_dst = os.path.join(val_labels_dir, label_filename)
    # 파일이 존재하면 이동
    if os.path.exists(label_src):
        shutil.move(label_src, label_dst)
        print(f"Moved {label_src} to {label_dst}")
    else:
        print(f"Label file {label_src} not found.")