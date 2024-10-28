import pandas as pd
import numpy as np
from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion
from pycocotools.coco import COCO
import os
import argparse

def main(fusion_method='nms', iou_thr=0.6, weights=None):
    # ensemble할 csv 파일들
    submission_files = [
        './csv/Co-DETR(Obj365, 1ep)_val.csv',
        './csv/Co-DETR(Obj365, 2ep)_val.csv',
        './csv/Co-DETR(Obj365, 3ep)_val.csv'
    ]

    # CSV 파일들을 DataFrame으로 읽어오기
    submission_df = [pd.read_csv(file) for file in submission_files]

    # 첫 번째 CSV 파일에서 이미지 ID 목록 가져오기
    image_ids = submission_df[0]['image_id'].tolist()

    # 테스트 데이터 JSON 파일 경로 설정
    annotation = '../dataset/json/test.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []

    # 각 이미지에 대해 앙상블 수행
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]

        # 각 모델(CSV 파일)의 예측 결과 처리
        for df in submission_df:
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            # 예측이 없는 경우 건너 뛰기
            if len(predict_list) == 0 or len(predict_list) == 1:
                continue

            # 예측 문자열을 numpy 배열로 변환
            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            # 박스 좌표를 이미지 크기에 맞게 정규화
            for box in predict_list[:, 2:6].tolist():
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 박스 좌표 앙상블 수행
        if len(boxes_list):
            if fusion_method == 'nms':
                boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr, weights=weights)
            elif fusion_method == 'soft_nms':
                boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr, weights=weights)
            elif fusion_method == 'nmw':
                boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, iou_thr=iou_thr, weights=weights)
            elif fusion_method == 'wbf':
                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr, weights=weights)
            else:
                raise ValueError("Invalid fusion method. Choose from 'nms', 'soft_nms', 'nmw', or 'wbf'.")

            # 클래스 레이블을 정수로 변환
            labels = [int(label) for label in labels]

            # 앙상블 결과를 문자열로 변환
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += f"{label} {score:.4f} {box[0]*image_info['width']:.2f} {box[1]*image_info['height']:.2f} {box[2]*image_info['width']:.2f} {box[3]*image_info['height']:.2f} "

        prediction_strings.append(prediction_string.strip())
        file_names.append(image_id)

    # 앙상블 결과를 DataFrame으로 저장
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    # 결과 저장
    os.makedirs('./output', exist_ok=True)
    output_file = f'./output/{fusion_method}_ensemble.csv'
    submission.to_csv(output_file, index=False, quoting=1, quotechar='"', escapechar='\\')
    print(f"Ensemble result saved to {output_file}")

if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description='Ensemble object detection results')
    # 앙상블 방법 선택
    parser.add_argument('--method', type=str, default='nms', choices=['nms', 'soft_nms', 'nmw', 'wbf'],
                        help='Fusion method to use (default: nms)')
    # IoU 임계값 설정
    parser.add_argument('--iou_thr', type=float, default=0.6,
                        help='IoU threshold for box fusion (default: 0.6)')
    # 각 모델에 대한 가중치 설정
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Weights for each model (default: None, which means equal weights)')
    args = parser.parse_args()

    # 앙상블 수행
    main(fusion_method=args.method, iou_thr=args.iou_thr, weights=args.weights)