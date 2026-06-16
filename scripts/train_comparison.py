from ultralytics import YOLO

# 세 버전의 데이터셋 설정
datasets = [
    {
        'name': 'no_aug',
        'data': '/home/ubuntu/dataset_no_aug/data.yaml',
        'description': '증강 없음'
    },
    {
        'name': 'basic_aug',
        'data': '/home/ubuntu/dataset_basic_aug/data.yaml',
        'description': '기본 증강'
    },
    {
        'name': 'strong_aug',
        'data': '/home/ubuntu/dataset_strong_aug/data.yaml',
        'description': '강한 증강'
    }
]

# YOLO 자체 증강 끄기 (Roboflow 증강만 효과 측정)
yolo_no_aug_settings = {
    'augment': False,
    'hsv_h': 0, 'hsv_s': 0, 'hsv_v': 0,
    'degrees': 0, 'translate': 0, 'scale': 0,
    'shear': 0, 'perspective': 0,
    'flipud': 0, 'fliplr': 0,
    'mosaic': 0, 'mixup': 0, 'copy_paste': 0,
}

# 동일한 조건으로 세 번 학습
for config in datasets:
    print(f"\n{'='*60}")
    print(f"학습 시작: {config['description']} ({config['name']})")
    print(f"{'='*60}\n")
    
    # 매번 새로운 모델 로드
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=config['data'],
        epochs=50,
        imgsz=640,
        batch=16,
        seed=42,
        name=f"baseball_{config['name']}",
        project='runs/comparison',
        **yolo_no_aug_settings
    )
    
    print(f"\n학습 완료: {config['name']}")

print("\n" + "="*60)
print("모든 학습 완료!")
print("="*60)
