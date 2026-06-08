import cv2
import os

VIDEO_PATH = "falling.mp4"
OUTPUT_DIR = "fall_frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 10)   # 초당 10장 저장

print("영상 FPS:", fps)
print("저장 간격:", frame_interval)

count = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if count % frame_interval == 0:
        cv2.imwrite(
            f"{OUTPUT_DIR}/fall_frame_{saved:04d}.jpg",
            frame
        )
        saved += 1

    count += 1

cap.release()
print("count =", count)
print("saved =", saved)
print(f"저장 완료: {saved}장")