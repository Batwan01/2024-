import os
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
from PIL import Image  # 이미지 크기를 얻기 위해 추가

import sys
sys.path.append('..')

def main():
    # 설정 파일 및 체크포인트 파일 경로
    config_name = 'co_dino_5scale_swin_l_16xb1_16e_o365tococo_custom_'
    config_file = f'../custom_configs/{config_name}.py'  # 모델 설정 파일 경로

    model_epoch = 1
    checkpoint_file = f'../checkpoints/best_coco_bbox_mAP_epoch_1.pth'  # 체크포인트 파일 경로

    # 이미지 경로 및 결과 저장 경로 설정
    image_folder = '../../tld_db/test/images'  # 이미지 폴더 경로
    output_folder = f'../output/predictions'  # YOLO 형식 출력 텍스트 파일 저장 폴더

    # 예측 결과 저장 폴더 생성 (없을 경우)
    os.makedirs(output_folder, exist_ok=True)

    # 모델 초기화
    print("Initializing model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 이미지 파일 목록 생성
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.jpg', '.png'))]

    # 이미지 추론
    print("Starting inference...")
    for image_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, image_name)

        # 이미지 크기 가져오기
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        result = inference_detector(model, img_path)

        # 결과 저장할 텍스트 파일 경로 설정
        txt_filename = os.path.splitext(image_name)[0] + ".txt"  # 이미지 이름을 사용한 텍스트 파일 이름
        txt_filepath = os.path.join(output_folder, txt_filename)

        # DetDataSample에서 결과 추출
        prediction_lines = []
        if hasattr(result, 'pred_instances'):
            det_samples = result.pred_instances  # 예측 결과의 인스턴스들
            if det_samples is not None:
                bboxes = det_samples.bboxes
                scores = det_samples.scores
                labels = det_samples.labels

                for j in range(len(bboxes)):
                    xmin, ymin, xmax, ymax = bboxes[j]

                    # YOLO 형식의 중심 좌표와 폭, 높이 계산
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height
                    confidence = scores[j]

                    # YOLO 형식으로 출력
                    prediction_lines.append(
                        f"{int(labels[j])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.4f}"
                    )

        # 텍스트 파일에 YOLO 형식의 예측 결과 저장
        with open(txt_filepath, 'w') as f:
            f.write("\n".join(prediction_lines))

    print(f"Inference complete. Predictions saved to {output_folder}")

if __name__ == '__main__':
    main()
