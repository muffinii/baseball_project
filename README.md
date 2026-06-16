# Baseball Swing Detection
YOLOv8 기반 야구 스윙 자동 판정 시스템

## 프로젝트 개요
야구 중계 영상에서 타자와 배트를 검출하고, 배트의 움직임을 분석하여 스윙 여부를 자동 판정하는 시스템

## 주요 기능
- YOLOv8 기반 객체 검출 (Batter, Bat)
- 배트 위치 추적
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

## 사용 방법

### 1. 모델 학습 (선택)
이미 학습된 모델(`models/best.pt`)을 제공
직접 학습하는 방법

#### 데이터셋 다운로드 (Roboflow API 키 필요)
python scripts/download_datasets.py

#### 증강 효과 비교 실험
python scripts/train_comparison.py

#### 모델 크기 비교 실험
python scripts/train_sizes.py

### 2. 스윙 판정 실행
python scripts/swing_detection.py
기본적으로 `test_video.mp4`를 분석하여 `output_swing.mp4`를 생성합니다.

### 3. 시스템 평가
여러 영상으로 시스템 성능 평가 방법

#### 테스트 영상 폴더 구조
```
test_videos/
├── swing/        # 스윙 영상
└── no_swing/     # 비스윙 영상
```

#### 평가 실행
python scripts/evaluate_system.py


## 프로젝트 구조
```
baseball-swing-detection/
├── scripts/              # 실행 스크립트
│   ├── extract_frames.py
│   ├── train_comparison.py
│   ├── train_sizes.py
│   ├── swing_detection.py
│   └── evaluate_system.py
├── best.pt               # 학습된 모델
├── results/              # 실험 결과
│   ├── comparison_results.csv
│   └── size_comparison_results.csv
└── README.md
```

## 실험 결과

### 데이터 증강 효과 비교
| Condition | mAP@0.5 | Precision | Recall |
|-----------|---------|-----------|--------|
| no_aug    | 0.9058  | 0.9466    | 0.8675 |
| **basic_aug** | **0.9220** | **0.9758** | **0.9067** |
| strong_aug | 0.9003 | 0.9543    | 0.8676 |

### 모델 크기 비교
| Model | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| v8n   | 0.9301  | 0.9386    | 0.9085 |
| **v8s** | **0.9514** | **0.9599** | **0.9274** |
| v8m   | 0.8582  | 0.9397    | 0.8314 |

### 최종 시스템 성능
- Accuracy: 90%
- F1 Score: 91%
- Test videos: 10 (5 swing, 5 no swing)