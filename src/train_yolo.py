"""
YOLOv8 Training Script
Trains a custom YOLOv8 model on your crop disease dataset
"""

from ultralytics import YOLO
import torch
import os

# ── CPU Training Config ─────────────────────────────────
# Optimized for 4-6 hour training without a dedicated GPU
DATASET_YAML = "dataset/multicrop_dataset.yaml"
MODEL_BASE   = "yolov8n.pt"      # Fastest, smallest model
OUTPUT_DIR   = "models/multicrop_run"
EPOCHS       = 15                # 15 epochs for good convergence on small dataset
IMG_SIZE     = 224               # Matches preprocessing resolution
BATCH_SIZE   = 16
WORKERS      = 4
FRACTION     = 1.0               # Use 100% of sampled dataset (only ~3k imgs)

def train():
    print("🚀 Starting YOLOv8 Training...")
    print(f"   Model   : {MODEL_BASE}")
    print(f"   Epochs  : {EPOCHS}")
    print(f"   Img Size: {IMG_SIZE}")
    print(f"   Batch   : {BATCH_SIZE}")
    print(f"   Workers : {WORKERS}")
    print()

    # Load pretrained YOLOv8 (downloads automatically first time)
    model = YOLO(MODEL_BASE)

    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,

        # 🔥 CPU Optimization constraints
        device="cpu",
        fraction=FRACTION,
        cache=False,  # Disabled caching to avoid RAM memory overflow on huge dataset
        augment=False,

        # 📁 Output
        project=OUTPUT_DIR,
        name="crop_disease_v1",
        exist_ok=True,

        # 📊 Basic training control
        patience=15,
        save=True,
        plots=True
    )

    print("\n✅ Training complete!")
    print(f"   Best model saved to: {OUTPUT_DIR}/crop_disease_v1/weights/best.pt")

    # Copy best model to models/
    import shutil
    shutil.copy(
        f"{OUTPUT_DIR}/crop_disease_v1/weights/best.pt",
        "models/yolov8_custom.pt"
    )
    print("   Copied to: models/yolov8_custom.pt")

    return results

def evaluate():
    print("\n📊 Evaluating model on test set...")
    model = YOLO("models/yolov8_custom.pt")
    metrics = model.val(data=DATASET_YAML, split="test")
    print(f"\n   mAP@0.5   : {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:95: {metrics.box.map:.4f}")
    print(f"   Precision : {metrics.box.p.mean():.4f}")
    print(f"   Recall    : {metrics.box.r.mean():.4f}")

if __name__ == "__main__":
    train()
    evaluate()