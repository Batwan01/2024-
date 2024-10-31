import pandas as pd
import numpy as np
import os

def filter_low_confidence(input_csv, output_csv, confidence_threshold):
    # CSV 파일 읽기
    df = pd.read_csv(input_csv)
    
    # 결과를 저장할 리스트
    filtered_predictions = []
    
    # 각 행을 순회하며 처리
    for _, row in df.iterrows():
        image_id = row['image_id']
        predictions = row['PredictionString'].split()
        
        # 예측이 없는 경우 빈 문자열 추가
        if len(predictions) == 0:
            filtered_predictions.append({'PredictionString': '', 'image_id': image_id})
            continue
        
        # 예측을 6개씩 그룹화 (클래스, 신뢰도, x1, y1, x2, y2)
        predictions = np.array(predictions).reshape(-1, 6)
        
        # confidence threshold보다 크거나 같은 예측만 필터링
        filtered = predictions[predictions[:, 1].astype(float) > confidence_threshold]
        
        # 필터링된 예측을 문자열로 변환
        filtered_string = ' '.join(filtered.reshape(-1))
        
        # 결과 추가 (PredictionString을 먼저 추가)
        filtered_predictions.append({'PredictionString': filtered_string, 'image_id': image_id})
    
    # 결과를 DataFrame으로 변환
    result_df = pd.DataFrame(filtered_predictions)
    
    # 열 순서를 원본과 동일하게 설정
    result_df = result_df[['PredictionString', 'image_id']]
    
    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_csv, index=False)
    print(f"Filtered predictions saved to {output_csv}")

if __name__ == "__main__":
    # 입력 및 출력 CSV 파일 경로 지정
    confidence_threshold = 0
    csv_name = 'nmw0.4(Cascade(2048, 2ep), Co-DETR(obj)1, 2, 3)(1, 0.8, 1.2, 1.5)'
    input_csv = f'./output/{csv_name}.csv'
    output_csv = f'./output/{csv_name}_{confidence_threshold}filtered.csv'
    

    # 출력 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    filter_low_confidence(input_csv, output_csv, confidence_threshold)