from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import os
import glob

# 모델 로드
model = YOLO('best.pt')

# 파라미터
WINDOW_SIZE = 3
MOVEMENT_THRESHOLD = 60


def detect_swing_in_video(video_path):
    """
    영상에서 스윙 여부 판정
    반환: True (스윙) 또는 False (비스윙)
    """
    cap = cv2.VideoCapture(video_path)
    bat_positions = deque(maxlen=WINDOW_SIZE)

    swing_detected = False  # 영상 중 한 번이라도 스윙 감지되면 True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        bat_center = None
        best_conf = 0

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if class_name == 'Bat' and confidence > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bat_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                best_conf = confidence

        if bat_center is not None:
            bat_positions.append(bat_center)

        # 스윙 판정
        if len(bat_positions) >= 3:
            total_movement = 0
            for i in range(1, len(bat_positions)):
                prev = bat_positions[i - 1]
                curr = bat_positions[i]
                dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
                total_movement += dist

            avg_movement = total_movement / len(bat_positions)

            if avg_movement > MOVEMENT_THRESHOLD:
                swing_detected = True
                break  # 스윙이 한 번이라도 감지되면 종료

    cap.release()
    return swing_detected


# 평가 실행
print("=" * 60)
print("스윙 판정 시스템 평가")
print("=" * 60)

# 스윙 영상 평가
swing_videos = glob.glob('test_videos/swing/*.mp4')
no_swing_videos = glob.glob('test_videos/no_swing/*.mp4')

print(f"\n스윙 영상 {len(swing_videos)}개")
print(f"비스윙 영상 {len(no_swing_videos)}개")

# 결과 저장
results = []

# 스윙 영상 (정답: True)
print("\n[스윙 영상 평가]")
for video in swing_videos:
    predicted = detect_swing_in_video(video)
    correct = predicted == True
    results.append({
        'video': os.path.basename(video),
        'actual': 'Swing',
        'predicted': 'Swing' if predicted else 'No Swing',
        'correct': correct
    })
    status = "✅" if correct else "❌"
    print(f"  {status} {os.path.basename(video)}: 예측={'Swing' if predicted else 'No Swing'}")

# 비스윙 영상 (정답: False)
print("\n[비스윙 영상 평가]")
for video in no_swing_videos:
    predicted = detect_swing_in_video(video)
    correct = predicted == False
    results.append({
        'video': os.path.basename(video),
        'actual': 'No Swing',
        'predicted': 'Swing' if predicted else 'No Swing',
        'correct': correct
    })
    status = "✅" if correct else "❌"
    print(f"  {status} {os.path.basename(video)}: 예측={'Swing' if predicted else 'No Swing'}")

# 통계 계산
total = len(results)
correct_count = sum(1 for r in results if r['correct'])

# Confusion Matrix 계산
TP = sum(1 for r in results if r['actual'] == 'Swing' and r['predicted'] == 'Swing')
TN = sum(1 for r in results if r['actual'] == 'No Swing' and r['predicted'] == 'No Swing')
FP = sum(1 for r in results if r['actual'] == 'No Swing' and r['predicted'] == 'Swing')
FN = sum(1 for r in results if r['actual'] == 'Swing' and r['predicted'] == 'No Swing')

accuracy = correct_count / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "=" * 60)
print("최종 결과")
print("=" * 60)
print(f"전체 영상: {total}개")
print(f"정확하게 판정: {correct_count}개")
print(f"\nAccuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

print("\n[Confusion Matrix]")
print(f"              예측: Swing  예측: No Swing")
print(f"실제: Swing      {TP:3d}          {FN:3d}")
print(f"실제: No Swing   {FP:3d}          {TN:3d}")

# CSV로 저장
import csv

with open('evaluation_results2.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['video', 'actual', 'predicted', 'correct'])
    writer.writeheader()
    writer.writerows(results)

print("\n상세 결과: evaluation_results.csv")