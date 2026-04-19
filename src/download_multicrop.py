"""
Multi-Crop Dataset Builder from local PlantVillage folders.
Uses the existing dataset/raw/PlantVillage data already on disk.
Samples MAX_PER_CLASS images per class, converts to YOLO format,
and writes disease_info.json for dynamic predictions.
"""

import os, json, random, re
from pathlib import Path
from PIL import Image

# ── Config ───────────────────────────────────────────────
RAW_DIR       = Path("dataset/raw/PlantVillage/PlantVillage")
OUT_DIR       = Path("dataset/processed_multicrop")
MAX_PER_CLASS = 300     # Up to 300 per class for good accuracy
IMG_SIZE      = (224, 224)
SPLIT         = (0.70, 0.20, 0.10)
SEED          = 42
random.seed(SEED)

# Disease info for every class — shown dynamically in the UI
DISEASE_META = {
    # TOMATO
    "Tomato_Bacterial_spot": {
        "severity": "Moderate",
        "description": "Xanthomonas vesicatoria causing dark water-soaked spots on leaves and fruit.",
        "treatment": "Apply copper-based bactericide weekly. Avoid overhead irrigation. Remove infected debris."
    },
    "Tomato_Early_blight": {
        "severity": "Moderate",
        "description": "Alternaria solani causing concentric ring 'target-board' lesions on older leaves.",
        "treatment": "Apply chlorothalonil or mancozeb every 7–10 days. Remove and destroy lower infected leaves. Rotate crops annually."
    },
    "Tomato_Late_blight": {
        "severity": "Severe",
        "description": "Phytophthora infestans — a destructive water mold spreading fast in cool, wet conditions.",
        "treatment": "Apply metalaxyl + mancozeb immediately. Destroy all infected plant material. Avoid wetting foliage."
    },
    "Tomato_Leaf_Mold": {
        "severity": "Mild",
        "description": "Passalora fulva causing olive-brown mold patches on leaf undersides in humid conditions.",
        "treatment": "Reduce humidity below 85%. Improve air circulation. Apply copper-based or chlorothalonil fungicide."
    },
    "Tomato_Septoria_leaf_spot": {
        "severity": "Moderate",
        "description": "Septoria lycopersici causing small circular spots with grey centers and dark borders.",
        "treatment": "Apply mancozeb or chlorothalonil. Remove infected leaves immediately. Mulch soil to reduce splash spread."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Moderate",
        "description": "Tetranychus urticae causing yellowing, stippling, and fine webbing on leaf undersides.",
        "treatment": "Apply miticide or neem oil. Increase ambient humidity. Introduce predatory mites (Phytoseiidae)."
    },
    "Tomato__Target_Spot": {
        "severity": "Moderate",
        "description": "Corynespora cassiicola causing brown circular lesions with concentric yellow-brown rings.",
        "treatment": "Apply azoxystrobin or difenoconazole. Improve air circulation. Remove lowest leaves as preventive measure."
    },
    "Tomato__Tomato_mosaic_virus": {
        "severity": "Severe",
        "description": "Tomato Mosaic Virus (ToMV) causing mosaic leaf patterns, distortion, and stunted growth.",
        "treatment": "Remove and destroy infected plants immediately. Control aphid vectors with insecticide. Disinfect all tools."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "severity": "Severe",
        "description": "TYLCV transmitted by whiteflies causing severe leaf curl, yellowing, and significant yield loss.",
        "treatment": "Control whitefly with imidacloprid. Use reflective mulches. Plant resistant varieties where available."
    },
    "Tomato_healthy": {
        "severity": "None",
        "description": "No disease detected. Tomato plant appears healthy with vibrant green foliage.",
        "treatment": "Continue balanced NPK fertilization. Scout weekly for early pest and disease signs."
    },
    # POTATO
    "Potato___Early_blight": {
        "severity": "Moderate",
        "description": "Alternaria solani causing dark brown concentric ring lesions on older potato leaves.",
        "treatment": "Apply chlorothalonil or mancozeb. Avoid overhead irrigation. Ensure good soil drainage."
    },
    "Potato___Late_blight": {
        "severity": "Severe",
        "description": "Phytophthora infestans causing water-soaked lesions rapidly turning brown-black; highly contagious.",
        "treatment": "Apply metalaxyl fungicide immediately. Destroy infected plants entirely. Never leave infected debris in field."
    },
    "Potato___healthy": {
        "severity": "None",
        "description": "No disease detected. Potato plant appears healthy.",
        "treatment": "Maintain consistent watering schedule and hilling. Scout weekly for early disease signs."
    },
    # PEPPER
    "Pepper__bell___Bacterial_spot": {
        "severity": "Moderate",
        "description": "Xanthomonas campestris causing water-soaked spots that turn brown with yellow halos on leaves.",
        "treatment": "Apply copper-based bactericide. Use disease-free certified seeds. Avoid working in field when wet."
    },
    "Pepper__bell___healthy": {
        "severity": "None",
        "description": "No disease detected. Bell pepper plant appears healthy.",
        "treatment": "Maintain regular watering. Apply balanced fertilizer. Monitor for pests weekly."
    },
}

def sanitize(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip('. ')

def make_yolo_label(class_idx):
    return f"{class_idx} 0.5 0.5 0.9 0.9\n"

def create_dirs():
    for split in ["train", "val", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print("Directories created")

def collect_class_data():
    classes = []
    all_items = []
    class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG")) + \
               list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.PNG"))
        if not imgs:
            continue
        # Cap per class
        sampled = random.sample(imgs, min(MAX_PER_CLASS, len(imgs)))
        class_idx = len(classes)
        classes.append(cls_name)
        for img_path in sampled:
            all_items.append((img_path, class_idx))
        print(f"  {cls_name}: {len(sampled)} images (class {class_idx})")

    return classes, all_items

def process_split(items, split_name, classes):
    img_dir = OUT_DIR / split_name / "images"
    lbl_dir = OUT_DIR / split_name / "labels"
    saved = 0
    for i, (img_path, class_idx) in enumerate(items):
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            stem = sanitize(img_path.stem) + f"_{i}"
            img.save(img_dir / f"{stem}.jpg", "JPEG", quality=90)
            (lbl_dir / f"{stem}.txt").write_text(make_yolo_label(class_idx))
            saved += 1
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
        if (saved % 300) == 0 and saved > 0:
            print(f"   {saved}/{len(items)} done")
    print(f"{split_name}: {saved}/{len(items)} images saved")

def make_yaml(classes):
    content = f"""# Multi-Crop YOLO Dataset (Tomato + Potato + Pepper)
path: E:\\crop_disease_detection\\dataset\\processed_multicrop
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    with open("dataset/multicrop_dataset.yaml", "w") as f:
        f.write(content)
    print("dataset/multicrop_dataset.yaml created")

def save_disease_meta(classes):
    out = {}
    for cls in classes:
        if cls in DISEASE_META:
            out[cls] = DISEASE_META[cls]
        else:
            parts = cls.replace("___", "||").replace("__", "||").split("||")
            crop    = parts[0].replace("_", " ").strip()
            disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else cls.replace("_", " ")
            is_healthy = "healthy" in disease.lower()
            out[cls] = {
                "severity": "None" if is_healthy else "Unknown",
                "description": f"{crop} plant appears healthy." if is_healthy else f"{disease} detected on {crop} plant.",
                "treatment": "Continue standard monitoring and balanced fertilization." if is_healthy
                             else f"Isolate affected plants. Consult an agronomist for a targeted {disease} treatment on {crop}."
            }
    Path("models").mkdir(exist_ok=True)
    with open("models/disease_info.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"models/disease_info.json saved ({len(out)} classes)")

if __name__ == "__main__":
    print("Building multi-crop dataset from local PlantVillage folders...")
    print(f"Source: {RAW_DIR}")

    classes, all_items = collect_class_data()

    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    print(f"\nTotal: {n} | Train: {n_train} | Val: {n_val} | Test: {n - n_train - n_val}")

    create_dirs()
    print("\nProcessing train split...")
    process_split(all_items[:n_train],          "train", classes)
    print("\nProcessing val split...")
    process_split(all_items[n_train:n_train+n_val], "val", classes)
    print("\nProcessing test split...")
    process_split(all_items[n_train+n_val:],    "test",  classes)

    make_yaml(classes)
    save_disease_meta(classes)
    print("\nMulti-crop dataset preparation complete!")
