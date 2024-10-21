from ultralytics import YOLO

def main():
    # 학습 설정
    data_yaml = '../tld_db/yaml/dataset_cropped.yaml'
    epochs = 20
    batch_size = 8
    img_size = 640
    device = '0'

    # 모델 로드
    model = YOLO('yolo11m.pt')

    # 학습 시작
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='CV Object Detection',
        name='ADAIC_yolo11m_cropped',
    )

    # 학습된 모델 저장
    # model.save('yolo11x_mixup.pt')

if __name__ == '__main__':
    main()