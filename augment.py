import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm
import albumentations as A

# ==========================================

# 설정

# ==========================================

SRC_DATASET = "data0608"
DST_DATASET = "data0608_augment"

# ==========================================

# 원본 복사

# ==========================================

print("원본 데이터셋 복사 중...")

for split in ["train", "valid", "test"]:


    os.makedirs(f"{DST_DATASET}/{split}/images", exist_ok=True)
    os.makedirs(f"{DST_DATASET}/{split}/labels", exist_ok=True)

    src_img = f"{SRC_DATASET}/{split}/images"
    src_lbl = f"{SRC_DATASET}/{split}/labels"

    dst_img = f"{DST_DATASET}/{split}/images"
    dst_lbl = f"{DST_DATASET}/{split}/labels"

for f in os.listdir(src_img):
    shutil.copy2(
        os.path.join(src_img, f),
        os.path.join(dst_img, f)
    )

for f in os.listdir(src_lbl):
    shutil.copy2(
        os.path.join(src_lbl, f),
        os.path.join(dst_lbl, f)
    )


print("원본 복사 완료")

# ==========================================

# Albumentations 설정

# ==========================================

bbox_params = A.BboxParams(
format="yolo",
label_fields=["labels"],
min_visibility=0.3
)

pipelines = [


A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=(0.1, 0.3),
        contrast_limit=0.1,
        p=1
    )
], bbox_params=bbox_params),

A.Compose([
    A.HorizontalFlip(p=1)
], bbox_params=bbox_params),

A.Compose([
    A.Rotate(
        limit=(-10, 10),
        border_mode=cv2.BORDER_CONSTANT,
        p=1
    )
], bbox_params=bbox_params),

A.Compose([
    A.MotionBlur(
        blur_limit=(5, 15),
        p=1
    )
], bbox_params=bbox_params)


]

# ==========================================

# 라벨 읽기 (깨진 라벨 자동 무시)

# ==========================================

def read_label(label_path):


    bboxes = []
    labels = []

    if not os.path.exists(label_path):
        return bboxes, labels

    with open(label_path, "r") as f:

        for line in f.readlines():

            p = line.strip().split()

            if len(p) != 5:
                continue

            try:

                cls = int(p[0])

                x = float(p[1])
                y = float(p[2])
                w = float(p[3])
                h = float(p[4])

                if w <= 0.001 or h <= 0.001:
                    continue

                if x - w/2 < 0:
                    continue

                if x + w/2 > 1:
                    continue

                if y - h/2 < 0:
                    continue

                if y + h/2 > 1:
                    continue

                bboxes.append([x, y, w, h])
                labels.append(cls)

            except:
                continue

    return bboxes, labels


# ==========================================

# 라벨 저장

# ==========================================

def save_label(path, bboxes, labels):


    with open(path, "w") as f:

        for label, bbox in zip(labels, bboxes):

            x, y, w, h = bbox

            f.write(
                f"{label} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


    # ==========================================

    # 증강 시작

    # ==========================================

    INPUT_IMG_DIR = f"{SRC_DATASET}/train/images"
    INPUT_LBL_DIR = f"{SRC_DATASET}/train/labels"

    OUTPUT_IMG_DIR = f"{DST_DATASET}/train/images"
    OUTPUT_LBL_DIR = f"{DST_DATASET}/train/labels"

    img_files = [
    f for f in os.listdir(INPUT_IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\n원본 train 이미지: {len(img_files)}장")
    print(f"증강 시작...\n")

    saved = 0
    skipped = 0

    for img_file in tqdm(img_files):


    stem = Path(img_file).stem
    ext = Path(img_file).suffix

    img_path = os.path.join(INPUT_IMG_DIR, img_file)
    lbl_path = os.path.join(INPUT_LBL_DIR, stem + ".txt")

    image = cv2.imread(img_path)

    if image is None:
        skipped += 1
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes, labels = read_label(lbl_path)

    if len(bboxes) == 0:
        skipped += 1
        continue

    for idx, pipeline in enumerate(pipelines):

        try:

            result = pipeline(
                image=image,
                bboxes=bboxes,
                labels=labels
            )

            aug_img = result["image"]
            aug_box = result["bboxes"]
            aug_lbl = result["labels"]

            if len(aug_box) == 0:
                continue

            out_img = os.path.join(
                OUTPUT_IMG_DIR,
                f"{stem}_aug{idx}{ext}"
            )

            out_lbl = os.path.join(
                OUTPUT_LBL_DIR,
                f"{stem}_aug{idx}.txt"
            )

            cv2.imwrite(
                out_img,
                cv2.cvtColor(
                    aug_img,
                    cv2.COLOR_RGB2BGR
                )
            )

            save_label(
                out_lbl,
                aug_box,
                aug_lbl
            )

            saved += 1

        except:
            continue


    print("\n==========================")
    print("증강 완료")
    print("==========================")
    print("생성:", saved)
    print("건너뜀:", skipped)

    train_img = len(os.listdir(f"{DST_DATASET}/train/images"))
    train_lbl = len(os.listdir(f"{DST_DATASET}/train/labels"))

    print(f"train images : {train_img}")
    print(f"train labels : {train_lbl}")
