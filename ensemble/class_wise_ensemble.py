import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from mAP50_calculator import calculate_map50

def get_class_aps(csv_path, gt_path):
    """각 CSV 파일의 클래스별 AP를 계산"""
    print(f"\nCalculating APs for {os.path.basename(csv_path)}")
    _, class_aps = calculate_map50(gt_path, csv_path)
    
    # 모든 클래스에 대해 AP 값을 가져오되, 없는 경우 0으로 처리
    return [class_aps.get(i, 0.0) for i in range(14)]  # 클래스 순서대로 AP 리스트 반환

def class_wise_ensemble():
    # ensemble할 csv 파일들
    submission_files = [
        './csv/Co-DETR(Obj365, 1ep)_val.csv',
        './csv/Co-DETR(Obj365, 2ep)_val.csv',
        './csv/Co-DETR(Obj365, 3ep)_val.csv',
        # './csv/Cascade-Rcnn(swinL, 2048, 2ep)_val.csv',
        # './csv/predictions_yolo11x_val.csv',
        # './csv/Cascade-Rcnn(swinL, 4096, 1ep)_val.csv',
        # './csv/Dino(1ep)_val.csv',
        # './csv/Co-DETR(SwinL, lsj, 3ep)_val.csv'
    ]

    print("Calculating class APs for each model...")
    gt_path = "./csv/val_ground_truth.csv"
    
    # 각 모델의 클래스별 AP 계산
    model_class_aps = []
    for file in submission_files:
        class_aps = get_class_aps(file, gt_path)
        model_class_aps.append(class_aps)
        
    # 각 클래스별로 가장 높은 AP를 가진 모델 선택
    class_best_model = {}
    CLASS_NAMES = ['veh_go', 'veh_goLeft', 'veh_noSign', 'veh_stop',
                   'veh_stopLeft', 'veh_stopWarning', 'veh_warning',
                   'ped_go', 'ped_noSign', 'ped_stop',
                   'bus_go', 'bus_noSign', 'bus_stop', 'bus_warning']
    
    for class_id in range(14):
        aps = [model_aps[class_id] for model_aps in model_class_aps]
        best_model_idx = np.argmax(aps)
        class_best_model[class_id] = best_model_idx
        print(f"Class {CLASS_NAMES[class_id]}: Best model = {os.path.basename(submission_files[best_model_idx])} (AP = {aps[best_model_idx]:.4f})")

    print("\nLoading CSV files...")
    submission_df = [pd.read_csv(file) for file in submission_files]
    image_ids = submission_df[0]['image_id'].tolist()

    prediction_strings = []
    file_names = []

    # 각 이미지에 대해 앙상블 수행
    print("\nPerforming class-wise best model ensemble...")
    for image_id in tqdm(image_ids, desc="Ensemble Progress"):
        prediction_string = ''
        
        # 각 모델의 예측을 클래스별로 수집
        for model_idx, df in enumerate(submission_df):
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            if pd.isna(predict_string):
                continue
                
            predict_list = str(predict_string).split()
            if len(predict_list) > 1:
                predict_list = np.reshape(predict_list, (-1, 6))
                for pred in predict_list:
                    class_id = int(pred[0])
                    if model_idx == class_best_model[class_id]:  # 해당 클래스에 대해 best인 모델의 예측만 선택
                        prediction_string += f"{' '.join(map(str, pred))} "

        prediction_strings.append(prediction_string.strip())
        file_names.append(image_id)

    # 앙상블 결과를 DataFrame으로 저장
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    # 결과 저장
    os.makedirs('./output', exist_ok=True)
    model_names = '_'.join([os.path.splitext(os.path.basename(f))[0].split('_val')[0] for f in submission_files])
    output_file = f'./output/class_wise_ensemble.csv'
    submission.to_csv(output_file, index=False)
    print(f"Ensemble result saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    # 앙상블 수행
    output_file = class_wise_ensemble()
    
    # 최종 mAP 50 계산
    print("\nCalculating final mAP50...")
    gt_path = "./csv/val_ground_truth.csv"
    mAP, _ = calculate_map50(gt_path, output_file)