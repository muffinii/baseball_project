from ultralytics import YOLO

models = [
    {'model': 'yolov8n.pt', 'name': 'v8n'},
    {'model': 'yolov8s.pt', 'name': 'v8s'},
    {'model': 'yolov8m.pt', 'name': 'v8m'},
]

yolo_no_aug_settings = {
    'augment': False,
    'hsv_h': 0, 'hsv_s': 0, 'hsv_v': 0,
    'degrees': 0, 'translate': 0, 'scale': 0,
    'shear': 0, 'perspective': 0,
    'flipud': 0, 'fliplr': 0,
    'mosaic': 0, 'mixup': 0, 'copy_paste': 0,
}

for cfg in models:
    print(f"\n학습 시작: {cfg['name']}")
    model = YOLO(cfg['model'])
    model.train(
        data='/home/ubuntu/dataset_basic_aug/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        seed=42,
        name=f"baseball_{cfg['name']}",
        project='runs/size_comparison'
    )

print("완료!")
