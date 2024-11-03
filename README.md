<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=2024%20자율주행%20인공지능%20챌린지&fontSize=50&animation=fadeIn&fontAlignY=38&desc=2024%20Autonomous%20Driving%20Artificial%20Intelligence%20Challenge&descAlignY=51&descAlign=62"/>
</p>

<!--
<div align="center">     
  <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/Batwan01/2024-Autonomous-Driving-Artificial-Intelligence-Challenge&count_bg=%23B8B8B8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
  <img src="https://img.shields.io/github/forks/2024-Autonomous-Driving-Artificial-Intelligence-Challenge" alt="forks"/>
  <img src="https://img.shields.io/github/stars/2024-Autonomous-Driving-Artificial-Intelligence-Challenge?color=yellow" alt="stars"/>
  <img src="https://img.shields.io/github/issues-pr/2024-Autonomous-Driving-Artificial-Intelligence-Challenge?color=red" alt="pr"/>
  <img src="https://img.shields.io/github/license/boostcamp-ai-tech-4/ai-tech-interview" alt="license"/>
</div>

!-->
---

## 💡 [프로젝트 소개](https://www.auto-dna.org/page/?M2_IDX=32625)

![car2](https://github.com/user-attachments/assets/6aa66e77-47f2-401d-a70f-773f433247aa)

- "자율주행 기술개발 혁신사업"을 통해 구축한 자율주행 공개 데이터셋을 활용하여 자율주행차-인프라 연계형 AI 기술 개발
- 주행환경에서 카메라 센서를 이용하여 신호등 인식

##  :sunglasses:팀원 소개

| [![](https://avatars.githubusercontent.com/jung0228)](https://github.com/jung0228) | [![](https://avatars.githubusercontent.com/chan-note)](https://github.com/chan-note) | [![](https://avatars.githubusercontent.com/batwan01)](https://github.com/batwan01) | [![](https://avatars.githubusercontent.com/jhuni17)](https://github.com/jhuni17) |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- |
| [정현우](https://github.com/jung0228)   |   [임찬혁](https://github.com/chan-note)     | [박지완](https://github.com/batwan01)          | [최재훈](https://github.com/jhuni17) |

## 데이터셋 정보

- **데이터셋 이름**: 신호등 데이터셋 ( 도로 상에 위치한 신호등을 인식하기 위한 학습 및 평가 데이터 )
- **출처**: [신호등 데이터셋 다운로드 링크](https://nanum.etri.re.kr/share/kimjy/TrafficLightAIchallenge2024?lang=ko_KR)
  
## 데이터셋 통계
- 학습 데이터: 26,864 프레임
- 평가 데이터: 13,505 프레임

![image](https://github.com/user-attachments/assets/2bc96bc8-a178-4581-b4d1-6687434c6593)

## 데이터 형태
  
| 신호등 유형   | 클래스 이름(번호)       | 신호등 유형   | 클래스 이름(번호)       |
|---------------|-------------------------|---------------|-------------------------|
| 차량 신호등   | Go (0)                 | 보행자 신호등 | Go (7)                  |
|               | GoLeft (1)             |               | NoSign (8)              |
|               | NoSign (2)             |               | Stop (9)                |
|               | Stop (3)               | 버스 신호등   | Go (10)                 |
|               | StopLeft (4)           |               | NoSign (11)             |
|               | StopWarning (5)        |               | Stop (12)               |
|               | Warning (6)            |               | Warning (13)            |
  
## Models

| Model | Backbone | Pre-trained | Epochs | oversampling | Image size | val mAP50 |
| --- | --- | --- | --- | --- | --- | --- |
| yolo | C3K2 | yolo11x | 20 | X | 1280x1280 | 0.6010 |
| Co-DINO | Swin-L | Object365,COCO | 1 | X | 1024x1024 | 0.6407 |
| Co-DINO | Swin-L | Object365,COCO | 2 | X | 1024x1024 | 0.6821 |
| Co-DINO | Swin-L | Object365,COCO | 3 | X | 1024x1024 | 0.6833 |
| Co-DINO | Swin-L | Object365,COCO | 1 | O | 1024x1024 | 0.6990 |
| Cascade-RCNN | Swin-L | COCO | 5 | X | 1024x1024 | 0.6819 |
| Cascade-RCNN | Swin-L | COCO | 2 | X | 1024x1024 | - |
| Cascade-RCNN | Swin-L | COCO | 2 | O | 1024x1024 | 0.6875 |




This project is released [OpenMMLab](https://github.com/open-mmlab) / [모델 성능표](https://github.com/Batwan01/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/issues/21)

## Ensemble

| 앙상블 기법 | Co-DINO 1ep over | Co-DINO 1ep | Co-DINO 2ep | Co-DINO 3ep | Cascade-RCNN 5ep | Cascade-RCNN 2ep  | Cascade-RCNN 2ep over | Test mAP50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NMW |  | o | o | o |  | o |  | 0.6945 |
| NMW | o |  |  | o | o |  | o | 0.7344 |
| Classwise | o | o | o | o | o |  | o | 0.7362 |

## Results

![image](https://github.com/user-attachments/assets/c739c8dc-16aa-4f40-80d2-900b74e33e52)

## How to use
