<p align='center'>
    <img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=2024%20자율주행%20인공지능%20챌린지&fontSize=50&animation=fadeIn&fontAlignY=38&desc=2024%20Autonomous%20Driving%20Artificial%20Intelligence%20Challenge&descAlignY=51&descAlign=62"/>
</p>

## 참고
- train 21533
- val 21534

| 커밋 유형 | 의미 |
| :-: | -|
|feat|	새로운 기능 추가|
|fix|	버그 수정|
|docs	|문서 수정|
|style|	코드 formatting, 세미콜론 누락, 코드 자체의 변경이 없는 경우|
|refactor	|코드 리팩토링|
|test|	테스트 코드, 리팩토링 테스트 코드 추가|
|chore|	패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore|
|design|	CSS 등 사용자 UI 디자인 변경|
|comment	|필요한 주석 추가 및 변경|
|rename|	파일 또는 폴더 명을 수정하거나 옮기는 작업만인 경우|
|remove|	파일을 삭제하는 작업만 수행한 경우|
|!BREAKING |CHANGE	커다란 API 변경의 경우|
|!HOTFIX	|급하게 치명적인 버그를 고쳐야 하는 경우|
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

![image](https://github.com/user-attachments/assets/cefe96f3-7780-4f05-941b-73594447ace4)

- "자율주행 기술개발 혁신사업"을 통해 구축한 자율주행 공개 데이터셋을 활용하여 자율주행차-인프라 연계형 AI 기술 개발 합니다.
- 객체 복합 상태 인식을 주 목표로 진행합니다.


## :clipboard: 일정
| 날짜 | 구분 |
| :-:| :-: |
| 8.27 ~ 9.30 | 신청 / 참가 |
| 10.02 ~ 11.01 | 대회 |
| 9.23 ~ 11.01 | 제출 및 평가 |
| 11.08 | 심사 결과 |
| 11월 중순 | 시상식 |

##  :sunglasses:팀원 소개

| [![](https://avatars.githubusercontent.com/jung0228)](https://github.com/jung0228) | [![](https://avatars.githubusercontent.com/chan-note)](https://github.com/chan-note) | [![](https://avatars.githubusercontent.com/batwan01)](https://github.com/batwan01) | [![](https://avatars.githubusercontent.com/jhuni17)](https://github.com/jhuni17) |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- |
| [정현우](https://github.com/jung0228)   |   [임찬혁](https://github.com/chan-note)     | [박지완](https://github.com/batwan01)          | [최재훈](https://github.com/jhuni17) |


## 개발 주제
| 주제 | 난이도 | 내용 |
| - | :-: | - |
| 3D객체 검출 | 최상 | 주행환경에서 라이다 센서를 이용하여 동적 객체(차량, 보행자, 자전거 이용자)를 검출 |
| 객체복합 상태인식 | 상 | 주행환경의 차량/버스를 인식하고 해당 객체의 의미론적(Semantic) 위치와<br> 후미등 상태를 인식하는 동시에 이미지 개체 분할(Instance Segmentation) |
| 3D객체 검출 | 중 | 인프라에서 수집된 연속된 카메라, 라이다의 융합된 프레임에서 3D 동적객체 검출 |
| 신호등 인식 | 하 | 주행환경에서 카메라 센서를 이용하여 신호등 인식 |

## 데이터셋 정보

- **데이터셋 이름**: 차량상태인식 ( 객체 복합상태인식 및 인스턴스 세그멘테이션 데이터셋 )
- **출처**: [차량상태인식 다운로드 링크](https://nanum.etri.re.kr/share/kimjy/ObjectStateDetectionAIchallenge2024?lang=ko_KR)
  
## 데이터셋 설명
- 자율주행 차량에서 전방의 차량, 버스를 2D-Bounding Box로 위치를 표현하고, 해당객체의 분류(Class), 의미론적 위치(Location), 후미등 상태(Action)를 분류하는 동시에 Instance Segmentation을 위한 인공지능 학습 데이터셋
- 자율주행 센서 데이터 수집차량을 이용한 수집 데이터( 제네시스 G80, IONIQ5, Carnival )

## 데이터 형태
- 데이터 파일

  - 전방 RBG 데이터 (.png)

  - 레이블 파일 (.txt)

  - 인스턴스 마스크 파일 (.png)

- 클래스 구분:  Class 2종, Location 5종, Action 4종
  
![e54f031e-c527-4c9b-afed-e4d98e44ed5c](https://github.com/user-attachments/assets/83af019f-9dfa-425d-aae8-9cddbd33e078)

## 데이터 통계
- 학습 데이터 / 평가 데이터

- 총 이미지 수 : 33,187 / 8,909

- 총 주석 객체 수: 243,919 / 미공개

  (차량: 228,864, 버스: 15,055) / 미공개

- 이미지당 평균 객체 수 : 7.35개
  
![35d3c259-e3e6-4811-a039-28a300b926da](https://github.com/user-attachments/assets/919d175d-de55-4170-9623-4a2c2da1d54d)
  
## Project TimeLine
| 순서 | 구분 |
| :-: | - |
| 1 | 기술 탐색 |
| 2 | 원리 이해( 논문 참고 ) |
| 3 | 코드 적용( git 참고 ) |
| 4 | 실험 |
| 5 | 결과 분석 |
| * | 코드 정리 |
| * | 문서 작성 |
| * | 코드&알고리즘 리뷰 |
| * | 재학습 Pipeline 구성 |
| * | 추론 인프라 구성 |

## Tools
- Github
- Slack
- Notion

## Models

## Backbones

## Augmentations

## Ensemble

## Results
