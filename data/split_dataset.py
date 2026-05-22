"""
클래스 비율 유지하면서 train/valid/test 나누기
- 각 클래스별로 비율 맞춰서 분리
- 기본: train 80%, valid 12%, test 8%
 
사용법:
  python split_dataset.py
"""
 
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
 
# ─── 설정 ───────────────────────────────────────────
INPUT_IMG_DIR  = "train/images"
INPUT_LBL_DIR  = "train/labels"
 
VALID_RATIO = 0.12  # 12%
TEST_RATIO  = 0.08  # 8%
# train은 나머지 80%
 
SEED = 42  # 재현성을 위한 시드
 
# ─── 출력 폴더 ───────────────────────────────────────
SPLITS = {
    "valid": ("valid/images", "valid/labels"),
    "test":  ("test/images",  "test/labels"),
}
 
# ─── 메인 ────────────────────────────────────────────
def main():
    random.seed(SEED)
 
    # 클래스별 파일 분류
    class_files = defaultdict(list)  # { class_id: [파일명, ...] }
    no_label    = []                 # 라벨 없는 파일
 
    img_files = [f for f in os.listdir(INPUT_IMG_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
 
    for img_file in img_files:
        stem     = Path(img_file).stem
        lbl_path = os.path.join(INPUT_LBL_DIR, stem + ".txt")
 
        if not os.path.exists(lbl_path):
            no_label.append(img_file)
            continue
 
        with open(lbl_path) as f:
            lines = f.readlines()
 
        if not lines:
            no_label.append(img_file)
            continue
 
        # 첫 번째 클래스 ID로 분류
        cls_id = int(lines[0].split()[0])
        class_files[cls_id].append(img_file)
 
    # 클래스 현황 출력
    print("=" * 50)
    print("클래스별 이미지 수:")
    for cls_id, files in sorted(class_files.items()):
        print(f"  클래스 {cls_id}: {len(files)}장")
    if no_label:
        print(f"  라벨 없음: {len(no_label)}장")
    print(f"  총합: {len(img_files)}장")
    print("=" * 50)
 
    # 출력 폴더 생성
    for split, (img_dir, lbl_dir) in SPLITS.items():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
 
    valid_files = []
    test_files  = []
 
    # 클래스별로 비율 맞춰서 분리
    for cls_id, files in sorted(class_files.items()):
        random.shuffle(files)
        total = len(files)
 
        n_valid = max(1, int(total * VALID_RATIO))
        n_test  = max(1, int(total * TEST_RATIO))
 
        valid_files.extend(files[:n_valid])
        test_files.extend(files[n_valid:n_valid + n_test])
 
        print(f"클래스 {cls_id}: train {total - n_valid - n_test}장 / valid {n_valid}장 / test {n_test}장")
 
    # 라벨 없는 파일도 비율 맞춰 분리
    if no_label:
        random.shuffle(no_label)
        n = len(no_label)
        n_valid = max(1, int(n * VALID_RATIO))
        n_test  = max(1, int(n * TEST_RATIO))
        valid_files.extend(no_label[:n_valid])
        test_files.extend(no_label[n_valid:n_valid + n_test])
 
    print("=" * 50)
 
    # 파일 이동
    def move_files(file_list, split):
        img_dir, lbl_dir = SPLITS[split]
        for img_file in file_list:
            stem = Path(img_file).stem
            ext  = Path(img_file).suffix
 
            src_img = os.path.join(INPUT_IMG_DIR, img_file)
            src_lbl = os.path.join(INPUT_LBL_DIR, stem + ".txt")
            dst_img = os.path.join(img_dir, img_file)
            dst_lbl = os.path.join(lbl_dir, stem + ".txt")
 
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
 
    move_files(valid_files, "valid")
    move_files(test_files,  "test")
 
    # 최종 확인
    print("\n✅ 분리 완료!")
    print(f"  train: {len(os.listdir(INPUT_IMG_DIR))}장")
    print(f"  valid: {len(os.listdir(SPLITS['valid'][0]))}장")
    print(f"  test:  {len(os.listdir(SPLITS['test'][0]))}장")
 
if __name__ == "__main__":
    main()