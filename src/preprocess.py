"""
Dataset Preparation Script
Converts PlantVillage folder structure → YOLO format
"""

import os
import re
import random
import json
from pathlib import Path
from PIL import Image

# ── Config ──────────────────────────────────────────────
RAW_DIR  = "dataset/raw/PlantVillage/PlantVillage"
OUT_DIR  = "dataset/processed"
SPLIT    = (0.70, 0.20, 0.10)
IMG_SIZE = 224
SEED     = 42
random.seed(SEED)

# ── Classes ──────────────────────────────────────────────
CLASSES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
]

def sanitize_filename(name):
    """Remove characters invalid in Windows filenames."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    return name

def create_dirs():
    for split in ["train", "val", "test"]:
        Path(f"{OUT_DIR}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUT_DIR}/{split}/yolo_images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUT_DIR}/{split}/labels").mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

def get_all_images():
    all_images = []
    for idx, cls in enumerate(CLASSES):
        cls_path = Path(RAW_DIR) / cls
        if not cls_path.exists():
            print(f"⚠️  Folder not found: {cls_path}")
            continue
        imgs = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.JPG")) + \
               list(cls_path.glob("*.png")) + list(cls_path.glob("*.PNG"))
        for img in imgs:
            all_images.append((img, idx))
        print(f"  {cls}: {len(imgs)} images (class {idx})")
    return all_images

def make_yolo_label(class_idx):
    cx, cy = 0.5, 0.5
    w,  h  = 0.9, 0.9
    return f"{class_idx} {cx} {cy} {w} {h}\n"

def split_and_copy(all_images):
    random.shuffle(all_images)
    n       = len(all_images)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    splits  = {
        "train": all_images[:n_train],
        "val"  : all_images[n_train:n_train + n_val],
        "test" : all_images[n_train + n_val:]
    }

    for split_name, items in splits.items():
        print(f"\n📁 Processing {split_name} ({len(items)} images)...")
        for img_path, class_idx in items:
            class_name = CLASSES[class_idx]

            # ── sanitize the filename ──────────────────
            safe_stem = sanitize_filename(img_path.stem)

            # ── DenseNet folder (class subfolders) ─────
            clf_dir = Path(f"{OUT_DIR}/{split_name}/images/{class_name}")
            clf_dir.mkdir(parents=True, exist_ok=True)

            # ── YOLO folders (flat) ────────────────────
            yolo_img_dir = Path(f"{OUT_DIR}/{split_name}/yolo_images")
            yolo_lbl_dir = Path(f"{OUT_DIR}/{split_name}/labels")

            try:
                img     = Image.open(img_path).convert("RGB")
                img_clf  = img.resize((224, 224))
                img_yolo = img.resize((IMG_SIZE, IMG_SIZE))

                # Save for DenseNet
                img_clf.save(clf_dir / f"{safe_stem}.jpg", "JPEG", quality=90)

                # Save for YOLO
                img_yolo.save(yolo_img_dir / f"{safe_stem}.jpg", "JPEG", quality=90)

                # Save YOLO label
                lbl_path = yolo_lbl_dir / f"{safe_stem}.txt"
                lbl_path.write_text(make_yolo_label(class_idx))

            except Exception as e:
                print(f"  ⚠️  Skipping {img_path.name}: {e}")
                continue

    print("\n✅ Split complete!")
    for s, items in splits.items():
        print(f"   {s}: {len(items)} images")

def make_yaml():
    yaml_content = f"""# YOLO Dataset Configuration
path: {os.path.abspath(OUT_DIR)}
train: train/yolo_images
val: val/yolo_images
test: test/yolo_images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open("dataset/dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("\n✅ dataset.yaml created")

def save_class_names():
    Path("models").mkdir(exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump({i: name for i, name in enumerate(CLASSES)}, f, indent=2)
    print("✅ class_names.json saved")

if __name__ == "__main__":
    print("🚀 Starting dataset preparation...")
    create_dirs()
    all_images = get_all_images()
    print(f"\n📊 Total images found: {len(all_images)}")
    split_and_copy(all_images)
    make_yaml()
    save_class_names()
    print("\n🎉 Dataset preparation complete!")