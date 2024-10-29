import os
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from tqdm import tqdm

import sys
sys.path.append('..')

def main():
    # 설정 파일 및 체크포인트 파일 경로
    config_name = 'deformable-detr-refine-twostage_r50_16xb2-50e_coco_trash'
    config_file = f'../custom_configs/{config_name}.py'  # 모델 설정 파일 경로
    checkpoint_file = f'../checkpoints/best_coco_bbox_mAP_epoch_1.pth'

    # 이미지 경로 및 결과 저장 경로 설정
    image_folder = '../../tld_db/test/images'  # 테스트 이미지 폴더 경로
    output_csv = f'../output/test_output.csv'  # 출력 CSV 파일 경로

    # 모델 초기화
    print("Initializing model...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')  # GPU 사용

    # 결과 저장 리스트
    results = []

    # 이미지 파일 목록 생성 (jpg, png 파일만 선택)
    image_files = [f for f in sorted(os.listdir(image_folder)) if f.endswith(('.jpg', '.png'))]

    # 이미지 추론
    print("Starting inference...")
    for image_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_folder, image_name)
        result = inference_detector(model, img_path)  # 모델을 사용하여 이미지에 대한 추론 수행

        # DetDataSample에서 결과 추출
        prediction_string = []
        if hasattr(result, 'pred_instances'):
            det_samples = result.pred_instances  # 예측 결과의 인스턴스들
            if det_samples is not None:
                bboxes = det_samples.bboxes  # 바운딩 박스 좌표
                scores = det_samples.scores  # 예측 신뢰도 점수
                labels = det_samples.labels  # 예측 클래스 레이블

                # 각 예측에 대한 문자열 생성
                for j in range(len(bboxes)):
                    prediction_string.append(
                        f"{int(labels[j])} {scores[j]:.4f} {bboxes[j][0]:.6f} {bboxes[j][1]:.6f} {bboxes[j][2]:.6f} {bboxes[j][3]:.6f}"
                    )

        # 결과를 리스트에 추가
        results.append({
            'PredictionString': " ".join(prediction_string),  # 모든 예측을 하나의 문자열로 결합
            'image_id': f'test/{image_name}'  # 이미지 ID 형식 지정
        })

    # 데이터프레임 생성
    print("Creating DataFrame...")
    df = pd.DataFrame(results)

    # CSV 파일로 저장
    print(f"Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    print(f"Inference complete. Results saved to {output_csv}")

if __name__ == '__main__':
    main()