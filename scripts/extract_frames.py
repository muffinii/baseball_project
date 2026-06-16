import cv2
import os
import numpy as np


def imwrite_korean(filename, img):
    try:
        ext = os.path.splitext(filename)[1]
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            with open(filename, mode='w+b') as f:
                encoded_img.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def imread_korean(filename):
    try:
        n = np.fromfile(filename, np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(e)
        return None


def extract_frames(video_path, output_folder, interval=10):
    os.makedirs(output_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"영상 열기 실패: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    video_name = os.path.basename(video_path).split('.')[0]

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % interval == 0:
            filename = f"{output_folder}/{video_name}_{saved_count:04d}.jpg"
            imwrite_korean(filename, frame)  # 한글 지원 함수 사용
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"{video_name}: {saved_count}장 추출 완료")


videos_folder = "videos"
frames_folder = "frames"

for video_file in os.listdir(videos_folder):
    if video_file.endswith(('.mp4', '.mkv', '.webm')):
        video_path = os.path.join(videos_folder, video_file)
        extract_frames(video_path, frames_folder, interval=10)