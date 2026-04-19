"""
Beans Dataset Downloader and Preprocessor
Downloads the Hugging Face 'beans' classification dataset and formats it for YOLOv8 Object Detection.
It uses a central bounding box strategy to convert classification images to object detection targets.
"""

from datasets import load_dataset
import os
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────
OUT_DIR = "dataset/processed_beans"
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]

def create_dirs():
    splits = ["train", "val", "test"]
    for split in splits:
        Path(f"{OUT_DIR}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUT_DIR}/{split}/labels").mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")

def make_yolo_label(class_idx):
    # Simulated bounding box covering 90% of the image (0.5x, 0.5y, 0.9w, 0.9h)
    return f"{class_idx} 0.5 0.5 0.9 0.9\n"

def process_and_save(dataset, split_name):
    print(f"\n📁 Processing {split_name} ({len(dataset)} images)...")
    img_dir = Path(f"{OUT_DIR}/{split_name}/images")
    lbl_dir = Path(f"{OUT_DIR}/{split_name}/labels")
    
    count = 0
    for i, item in enumerate(dataset):
        image = item["image"]
        label_idx = item["labels"]
        
        # Some images might not be RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Resize to 224 to keep the dataset extraordinarily fast and lightweight
        image = image.resize((224, 224))
        
        safe_stem = f"beans_{split_name}_{i}"
        
        # Save image
        image.save(img_dir / f"{safe_stem}.jpg", "JPEG", quality=90)
        
        # Save label
        lbl_path = lbl_dir / f"{safe_stem}.txt"
        lbl_path.write_text(make_yolo_label(label_idx))
        
        count += 1
        if count % 100 == 0:
            print(f"   Processed {count} / {len(dataset)}")
            
    print(f"✅ {split_name} split complete!")

def make_yaml():
    yaml_content = f"""# YOLO Beans Dataset Configuration
path: processed_beans
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open("dataset/beans_dataset.yaml", "w") as f:
        f.write(yaml_content)
    print("\n✅ dataset/beans_dataset.yaml created")

if __name__ == "__main__":
    print("🚀 Downloading the Hugging Face 'beans' dataset...")
    # 'beans' dataset has train, validation, and test splits
    dataset = load_dataset("beans")
    
    create_dirs()
    
    # Validation split is natively called 'validation'
    process_and_save(dataset["train"], "train")
    process_and_save(dataset["validation"], "val")
    process_and_save(dataset["test"], "test")
    
    make_yaml()
    print("\n🎉 Beans dataset preparation complete and ready for CPU training!")
