import os
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm
import argparse

import sys
sys.path.append('..')

def main():
    # 설정 파일 및 체크포인트 파일 경로
    config_name = args.config_name
    config_file = f'../custom_configs/{config_name}.py'  # 모델 설정 파일 경로
    checkpoint_file = f"../checkpoints/{config_name}.pth"

    # 이미지 경로 및 결과 저장 경로 설정
    image_folder = '../../tld_db/train/images'  # train 데이터 폴더 경로로 변경
    output_csv = f'../../post_processing/csv/{config_name}_val_inference.csv'

    # 모델 초기화
    print("Initializing model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # 결과 저장 리스트
    results = []

    # 이미지 파일 목록 생성 (21534번부터의 이미지만 선택)
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.jpg', '.png'))]
    val_images = [img for img in image_files if int(img.split('.')[0]) >= 21534]  # 21534부터의 이미지만 선택

    # 이미지 추론
    print("Starting inference...")
    for image_name in tqdm(val_images, desc="Processing validation images"):
        img_path = os.path.join(image_folder, image_name)
        result = inference_detector(model, img_path)

        # DetDataSample에서 결과 추출
        prediction_string = []
        if hasattr(result, 'pred_instances'):
            det_samples = result.pred_instances
            if det_samples is not None:
                bboxes = det_samples.bboxes
                scores = det_samples.scores
                labels = det_samples.labels

                for j in range(len(bboxes)):
                    prediction_string.append(
                        f"{int(labels[j])} {scores[j]:.4f} {bboxes[j][0]:.2f} {bboxes[j][1]:.2f} {bboxes[j][2]:.2f} {bboxes[j][3]:.2f}"
                    )

        # 결과를 리스트에 추가
        results.append({
            'PredictionString': " ".join(prediction_string),
            'image_id': image_name  # 이미지 이름만 저장
        })

    # 데이터프레임 생성
    print("Creating DataFrame...")
    df = pd.DataFrame(results)

    # CSV 파일로 저장
    print(f"Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Validation inference complete. Results saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', help='train config file path')

    args = parser.parse_args()
    main()