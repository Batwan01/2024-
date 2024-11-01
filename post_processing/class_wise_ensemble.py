import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from mAP50_calculator import calculate_map50

def get_class_aps(csv_path, gt_path):
    """각 CSV 파일의 클래스별 AP를 계산"""
    print(f"\nCalculating APs for {os.path.basename(csv_path)}")
    _, class_aps = calculate_map50(gt_path, csv_path)
    return [class_aps.get(i, 0.0) for i in range(14)]

def get_model_files():
    """앙상블할 validation과 test 파일 경로 반환"""
    val_files = [
        './csv/Co-DETR(Obj365, 1ep)_val.csv',
        './csv/Co-DETR(Obj365, 2ep)_val.csv',
        './csv/Co-DETR(Obj365, 3ep)_val.csv',
        './csv/Cascade-Rcnn(swinL, 2048, 5ep)_val.csv',
        './csv/Cascade-Rcnn(swinL, 2048, 2ep, oversampling)_val.csv',
        './csv/Co-DETR(Obj365, 2024, oversampling, ep1)_val.csv',
    ]
    
    test_files = [
        './csv/Co-DETR(Obj365, 1ep)_test.csv',
        './csv/Co-DETR(Obj365, 2ep)_test.csv',
        './csv/Co-DETR(Obj365, 3ep)_test.csv',
        './csv/Cascade-Rcnn(swinL, 2048, 5ep)_test.csv',
        './csv/Cascade-Rcnn(swinL, 2048, 2ep, oversampling)_test.csv',
        './csv/Co-DETR(Obj365, 2024, oversampling, ep1)_test.csv',
    ]
    
    return val_files, test_files

def get_best_models(val_files, gt_path):
    """validation 데이터로 클래스별 베스트 모델 선정"""
    print("Calculating class APs for each model...")
    
    model_class_aps = []
    for file in val_files:
        class_aps = get_class_aps(file, gt_path)
        model_class_aps.append(class_aps)
        
    class_best_model = {}
    CLASS_NAMES = ['veh_go', 'veh_goLeft', 'veh_noSign', 'veh_stop',
                   'veh_stopLeft', 'veh_stopWarning', 'veh_warning',
                   'ped_go', 'ped_noSign', 'ped_stop',
                   'bus_go', 'bus_noSign', 'bus_stop', 'bus_warning']
    
    for class_id in range(14):
        aps = [model_aps[class_id] for model_aps in model_class_aps]
        best_model_idx = np.argmax(aps)
        class_best_model[class_id] = best_model_idx
        print(f"Class {CLASS_NAMES[class_id]}: Best model = {os.path.basename(val_files[best_model_idx])} (AP = {aps[best_model_idx]:.4f})")
    
    return class_best_model

def ensemble_predictions(files, class_best_model, output_suffix):
    """주어진 파일들에 대해 앙상블 수행"""
    print(f"\nLoading {output_suffix} CSV files...")
    submission_df = [pd.read_csv(file) for file in files]
    image_ids = submission_df[0]['image_id'].tolist()

    prediction_strings = []
    file_names = []

    print(f"\nPerforming class-wise best model ensemble for {output_suffix}...")
    for image_id in tqdm(image_ids, desc="Ensemble Progress"):
        prediction_string = ''
        
        for model_idx, df in enumerate(submission_df):
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            if pd.isna(predict_string):
                continue
                
            predict_list = str(predict_string).split()
            if len(predict_list) > 1:
                predict_list = np.reshape(predict_list, (-1, 6))
                for pred in predict_list:
                    class_id = int(pred[0])
                    if model_idx == class_best_model[class_id]:
                        prediction_string += f"{' '.join(map(str, pred))} "

        prediction_strings.append(prediction_string.strip())
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names

    os.makedirs('./output', exist_ok=True)
    output_file = f'./output/class_wise_ensemble_{output_suffix}.csv'
    submission.to_csv(output_file, index=False)
    print(f"Ensemble result saved to {output_file}")
    
    return output_file

def class_wise_ensemble(mode='val'):
    """
    클래스별 앙상블 수행
    mode: 'val' (validation만), 'test' (validation으로 베스트모델 선정 후 test), 'both' (둘 다 수행)
    """
    val_files, test_files = get_model_files()
    gt_path = "./csv/val_ground_truth.csv"
    
    # Validation으로 베스트 모델 선정
    class_best_model = get_best_models(val_files, gt_path)
    
    output_files = []
    
    # 모드에 따라 앙상블 수행
    if mode in ['val', 'both']:
        val_output = ensemble_predictions(val_files, class_best_model, 'val')
        output_files.append(val_output)
        
        # validation mAP 계산
        print("\nCalculating validation mAP50...")
        mAP, _ = calculate_map50(gt_path, val_output)
        print(f"Final validation mAP50: {mAP:.4f}")
    
    if mode in ['test', 'both']:
        test_output = ensemble_predictions(test_files, class_best_model, 'test')
        output_files.append(test_output)
    
    return output_files

if __name__ == "__main__":
    # 모드 선택: 'val', 'test', 'both'
    output_files = class_wise_ensemble(mode='both')