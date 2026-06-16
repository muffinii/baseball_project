import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace("-vlfha").project("baseball-ioybu")

# 버전 1: 증강 없음
print("버전 1 다운로드 시작")
dataset_v1 = project.version(2).download('yolov8', location='./dataset_no_aug')
print("버전 1 완료")

# 버전 2: 기본 증강
print("버전 2 다운로드 시작")
dataset_v2 = project.version(1).download('yolov8', location='./dataset_basic_aug')
print("버전 2 완료")

# 버전 3: 강한 증강
print("버전 3 다운로드 시작")
dataset_v3 = project.version(3).download('yolov8', location='./dataset_strong_aug')
print("버전 3 완료")

print("\n세 버전 모두 다운로드 완료")
