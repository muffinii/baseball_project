from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# 모델 로드
model = YOLO('best.pt')

# 영상 로드
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

# 출력 영상 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_swing.mp4', fourcc, fps, (width, height))

# 배트 위치 기록용 (최근 3프레임)
WINDOW_SIZE = 3
bat_positions = deque(maxlen=WINDOW_SIZE)

# 스윙 판정 임계값
MOVEMENT_THRESHOLD = 60  # 픽셀 단위

frame_count = 0
swing_detected_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 검출
    results = model(frame, verbose=False)

    # 배트 위치 추출 (가장 신뢰도 높은 배트 1개만)
    bat_center = None
    best_conf = 0

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Bat 클래스이고 신뢰도가 더 높으면
        if class_name == 'Bat' and confidence > best_conf:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bat_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            best_conf = confidence

    # 배트 위치 기록 (못 찾으면 이전 위치 사용 또는 None)
    if bat_center is not None:
        bat_positions.append(bat_center)

    # 스윙 판정: 최근 프레임에서 배트 이동량 계산
    is_swing = False
    movement = 0

    if len(bat_positions) >= 3:
        total_movement = 0
        for i in range(1, len(bat_positions)):
            prev = bat_positions[i - 1]
            curr = bat_positions[i]
            dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            total_movement += dist

        movement = total_movement

        avg_movement = movement / len(bat_positions)

        if avg_movement > MOVEMENT_THRESHOLD:
            is_swing = True
            swing_detected_frames.append(frame_count)

    # 결과를 프레임에 그리기
    annotated_frame = results[0].plot()

    # 판정 결과 텍스트 추가
    status = "SWING!" if is_swing else "No Swing"
    color = (0, 0, 255) if is_swing else (0, 255, 0)
    cv2.putText(annotated_frame, status, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(annotated_frame, f"Movement: {movement:.1f}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(annotated_frame)
    frame_count += 1

    if frame_count % 30 == 0:
        print(f"처리 중: {frame_count} 프레임")

cap.release()
out.release()

print(f"\n완료! 총 {frame_count} 프레임")
print(f"스윙 감지된 프레임 수: {len(swing_detected_frames)}")
print("결과: output_swing.mp4")