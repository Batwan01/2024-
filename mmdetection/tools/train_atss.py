import os
import logging
import torch
import heapq
from mmcv import Config
# from mmengine import Config
from mmdet.apis import train_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device
from mmcv.runner import Hook
from mmcv.runner import HOOKS


@HOOKS.register_module()
class SaveTopModelsHook(Hook):
    """ 상위 3개 accuracy 및 loss 기준으로 모델을 저장하는 Hook """
    def __init__(self, work_dir):
        self.top3_val_acc = []  # accuracy 기준 상위 3개
        self.top3_val_loss = []  # loss 기준 상위 3개
        self.work_dir = work_dir

        

        # Logger 설정
        log_path = os.path.join(work_dir, 'train.log')
        logging.basicConfig(
            level=logging.INFO,  # 로그 레벨을 INFO로 설정
            format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 형식
            handlers=[
                logging.FileHandler(log_path),  # 로그를 파일에 기록
                logging.StreamHandler()  # 로그를 콘솔에도 출력
            ]
        )
        self.logger = logging.getLogger()

    def extract_last_bbox_map_50(self, log_file):
        last_bbox_map_50 = None

        with open(log_file, 'r') as file:
            lines = file.readlines()  # 모든 줄을 읽어옵니다.
        
        # 마지막 줄에서 값을 추출하기 위해 역순으로 반복
        for line in reversed(lines):
            if 'bbox_mAP_50' in line:
                # 줄에서 bbox_mAP_50 값을 정규 표현식을 사용하여 추출
                parts = line.split(',')
                for part in parts:
                    if 'bbox_mAP_50' in part:
                        # '=' 이후의 값을 가져오기
                        last_bbox_map_50 = float(part.split(':')[1].strip())
                        break
                # 값을 찾으면 반복 종료
                break

        return last_bbox_map_50
    
    def save_top_model(self, metric_list, value, epoch, filename, mode='max'):
        """ 상위 3개 모델을 저장하고 관리하는 함수 """
        if mode == 'max':
            # 상위 3개 accuracy를 추적
            heapq.heappush(metric_list, (value, epoch, filename))
            if len(metric_list) > 2:
                # 상위 3개만 유지하고, 가장 낮은 값을 제거
                old_model = heapq.heappop(metric_list)
                os.remove(old_model[2])  # 파일 삭제
        else:
            # 상위 3개 loss를 추적 (loss는 작을수록 좋음)
            heapq.heappush(metric_list, (-value, epoch, filename))
            if len(metric_list) > 2:
                # 상위 3개만 유지하고, 가장 큰 값을 제거
                old_model = heapq.heappop(metric_list)
                os.remove(old_model[2])  # 파일 삭제

    def after_train_epoch(self, runner):
        # validation accuracy, loss, mAP50을 얻음
        self.logger.info("After val epoch hook called.")
        path = os.path.join(self.work_dir, 'train.log')
        mAP50 = self.extract_last_bbox_map_50(path)  # mAP50 값
        epoch = runner.epoch + 1

        checkpoint_filename = os.path.join(self.work_dir, f"_mAP50_{mAP50:.4f}_epoch_{epoch}_.pth")

        # 전체 모델과 optimizer 상태를 포함한 체크포인트 저장
        torch.save({
        'state_dict': runner.model.state_dict(),  # 모델 가중치
        'optimizer': runner.optimizer.state_dict()  # 옵티마이저 상태
        }, checkpoint_filename)

        # 정확도 기준 상위 3개 가중치만 유지
        self.save_top_model(self.top3_val_acc, mAP50, epoch, checkpoint_filename, mode='max')


    def after_run(self, runner):
        # 학습 종료 후 필요한 정리 작업 수행
        self.logger.info("Training has ended.")


def main():
    """
    train_detector를 사용하여 모델을 훈련하는 메인 함수.
    """

    torch.cuda.set_device(1)
    # device = torch.device(f'cuda:{1}')
    print(f"is_available cuda : {torch.cuda.is_available()}")
    print(f"current use : cuda({torch.cuda.current_device()})\n")

    # Config 파일 로드 및 수정
    cfg = Config.fromfile("/home/hwang/leem/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/mmdetection/configs/dyhead/atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco.py")  # 모델 설정
    cfg.work_dir = './work_dirs/atss_sinhodeung'                      # 로그/모델 저장 위치
    # cfg.optimizer.type = 'SGD'                                                     # optimizer 설정
    # cfg.optimizer.lr = 0.02                                                        # lr 설정
    # cfg.optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)

    # cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)                # gradient clipping 설정
    cfg.data.samples_per_gpu = 2                                                 # 배치 크기 설정
    # cfg.runner = dict(type='EpochBasedRunner', max_epochs=18)                      # epoch 수 설정
    cfg.seed = 2022                                                                # 랜덤 시드 설정
    cfg.gpu_ids = [1]                                                              # 사용할 GPU 설정
    cfg.device = 'cuda'                                                  # 디바이스 설정 (GPU 또는 CPU)
    cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)

    # 내가한거임
    cfg.resume_from = None
    cfg.workflow = [('train', 1), ('val', 1)]

    cfg.visualizer.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'), # wandb 시각화, 사용시 wandb 계정 필요, 사용하지 않을 시 주석처리
    ]


    # TensorBoard 로그 및 텍스트 로그 설정
    cfg.log_config = dict(
        interval=50,  # 로그 기록 간격 (배치 단위)
        hooks=[
            dict(type='TextLoggerHook'),  # 텍스트 기반 로그 출력 (콘솔용)
            dict(type='TensorboardLoggerHook')  # TensorBoard 로그 기록
        ]
    )

    # 커스텀 Hook 추가 (설정 파일에 등록)
    cfg.custom_hooks = [dict(type='SaveTopModelsHook', work_dir=cfg.work_dir, priority='VERY_LOW')]

    # Dataset 설정
    classes = ("veh_go", "veh_goLeft", "veh_noSign", "veh_stop", "veh_stopLeft", 
               "vef_stopWarning", "veh_warning", "ped_go", "ped_noSign", "ped_stop", "bus_go", "bus_noSign", "bus_stop", "bus_warning")

    root = '/hdd1/lim_data/tld_db'  # root 경로 설정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = "/hdd1/lim_data/tld_db/json/train_coco.json"
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = "/hdd1/lim_data/tld_db/json/val_coco.json"

    # Train과 Val 데이터셋 각각 빌드
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.val)]

    # 모델 정의
    model = build_detector(cfg.model)
    model.init_weights()

    # work_dir 생성
    work_dir = cfg.work_dir
    os.makedirs(work_dir, exist_ok=True)

    # 모델 학습 시작
    train_detector(
        model=model,
        dataset=datasets,
        cfg=cfg,
        distributed=False,
        validate= True  # 검증 포함
    )


if __name__ == '__main__':
    main()
