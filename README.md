# Baseball Swing Detection
YOLOv8 기반 야구 스윙 자동 판정 시스템

## 프로젝트 개요
야구 중계 영상에서 타자와 배트를 검출하고, 배트의 움직임을 분석하여 스윙 여부를 자동 판정하는 시스템

## 주요 기능
- YOLOv8 기반 객체 검출 (Batter, Bat)
- 시계열 배트 위치 추적
- 이동량 기반 스윙 판정
- 다양한 증강 조건 비교 실험
- 모델 크기별 성능 비교

## 기술 스택
- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Roboflow (데이터 라벨링)
- AWS EC2 (학습 환경)

## 성능
- 객체 검출: mAP@0.5 = 0.9514 (YOLOv8s)
- 스윙 판정: Accuracy = 90%, F1 Score = 90.91%
