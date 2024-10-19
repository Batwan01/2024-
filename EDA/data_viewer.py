import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# 클래스 로드 함수
@st.cache_data
def load_classes(classes_path):
    with open(classes_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# YOLO 형식의 바운딩 박스를 (x_min, y_min, x_max, y_max) 형식으로 변환
def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x_min = int((x_center - width/2) * img_width)
    y_min = int((y_center - height/2) * img_height)
    x_max = int((x_center + width/2) * img_width)
    y_max = int((y_center + height/2) * img_height)
    return x_min, y_min, x_max, y_max

# 이미지에 바운딩 박스 그리는 함수
def draw_boxes(image, boxes, labels, classes):
    for box, label_idx in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        label = classes[label_idx]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Streamlit 앱 시작
st.title("Traffic Light Detection Visualization")

# 데이터 경로 설정
dataset_path = "../ultralytics/datasets/TLD_2024/tld_db"
train_path = os.path.join(dataset_path, 'train')
image_dir = os.path.join(train_path, 'images')
label_dir = os.path.join(train_path, 'labels')
classes_path = os.path.join(train_path, 'classes.txt')

# 클래스 로드
classes = load_classes(classes_path)

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 이미지 선택 슬라이더
image_index = st.slider("Select an image", 0, len(image_files) - 1, 0)
selected_image = image_files[image_index]

# 이미지 로드
image_path = os.path.join(image_dir, selected_image)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_height, img_width = image.shape[:2]

# 레이블 파일 로드
label_path = os.path.join(label_dir, os.path.splitext(selected_image)[0] + '.txt')
boxes = []
labels = []

if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            label, x_center, y_center, width, height = map(float, line.strip().split())
            box = yolo_to_bbox(x_center, y_center, width, height, img_width, img_height)
            boxes.append(box)
            labels.append(int(label))

# 바운딩 박스 그리기
image_with_boxes = draw_boxes(image.copy(), boxes, labels, classes)

# 이미지 표시
st.image(image_with_boxes, caption=f"Image: {selected_image}", use_column_width=True)

# 바운딩 박스 정보 표시
st.write("Bounding Box Information:")
for box, label in zip(boxes, labels):
    st.write(f"Class: {classes[label]}, Box: {box}")