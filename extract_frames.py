# 동영상 프레임 자르기

import cv2
import os

VIDEO_PATH = "dataset1.mp4"
OUTPUT_DIR = "frames"
INTERVAL_MS = 300  # 300ms마다 추출

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상 열기 실패")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * (INTERVAL_MS / 1000))

print(f"FPS: {fps}")
print(f"frame_interval: {frame_interval}")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:

    #     # 스마트폰 세로영상 회전
    #     frame = cv2.rotate(
    #     frame,
    #     cv2.ROTATE_90_CLOCKWISE
    # )

        # 라벨링용 크기 축소 (세로 비율 유지)
        # frame = cv2.resize(
        #     frame,
        #     (720, 1280)
        # )

        save_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(save_path, frame)

        saved_count += 1

        if saved_count % 50 == 0:
            print(f"{saved_count}장 저장 완료")

    frame_count += 1

cap.release()

print(f"\n총 {saved_count}장 저장 완료")
print(f"저장 위치: {OUTPUT_DIR}")