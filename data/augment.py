"""
YOLO 데이터셋 Augmentation 스크립트
- 200장 → 1400장 (1장당 7가지 변형)
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
import albumentations as A

INPUT_IMG_DIR  = "train/images"
INPUT_LBL_DIR  = "train/labels"
OUTPUT_IMG_DIR = "output/images"
OUTPUT_LBL_DIR = "output/labels"

AUGMENT_PER_IMAGE = 7  # 200장 × 7 = 1400장

pipelines = [
    # 1. 밝기 어둡게
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),

    # 2. 밝기 밝게
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.2, 0.4), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),

    # 3. 좌우 반전
    A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),

    # 4. 상하 반전
    A.Compose([
        A.VerticalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),

    # 5. 회전
    A.Compose([
        A.Rotate(limit=(-30, 30), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),

    # 6. 블러
    A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),

    # 7. 색조 변환
    A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_visibility=0.3)),
]

def read_label(label_path):
    bboxes = []
    labels = []
    if not os.path.exists(label_path):
        return bboxes, labels
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append(int(parts[0]))
            coords = [max(0.0, min(1.0, float(x))) for x in parts[1:5]]
            bboxes.append(coords)
    return bboxes, labels

def save_label(label_path, bboxes, labels):
    with open(label_path, 'w') as f:
        for label, bbox in zip(labels, bboxes):
            f.write(f"{label} {' '.join(f'{x:.6f}' for x in bbox)}\n")

def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

    img_files = [f for f in os.listdir(INPUT_IMG_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"원본 이미지: {len(img_files)}장")
    print(f"증강 후 예상: {len(img_files) * AUGMENT_PER_IMAGE}장")

    total_saved = 0

    for img_file in tqdm(img_files, desc="증강 중"):
        img_path = os.path.join(INPUT_IMG_DIR, img_file)
        stem     = Path(img_file).stem
        ext      = Path(img_file).suffix
        lbl_path = os.path.join(INPUT_LBL_DIR, stem + ".txt")

        image          = cv2.imread(img_path)
        image          = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, labels = read_label(lbl_path)

        for idx, pipeline in enumerate(pipelines):
            try:
                result     = pipeline(image=image, bboxes=bboxes, labels=labels)
                aug_img    = result['image']
                aug_bboxes = result['bboxes']
                aug_labels = result['labels']

                out_name = f"{stem}_aug{idx}{ext}"
                out_img  = os.path.join(OUTPUT_IMG_DIR, out_name)
                out_lbl  = os.path.join(OUTPUT_LBL_DIR, f"{stem}_aug{idx}.txt")

                cv2.imwrite(out_img, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                save_label(out_lbl, aug_bboxes, aug_labels)
                total_saved += 1

            except Exception as e:
                print(f"\n오류 ({img_file} aug{idx}): {e}")
                continue

    print(f"\n✅ 완료! 총 {total_saved}장 저장됨")

if __name__ == "__main__":
    main()
