# 🌿 CropGuard AI — Crop Disease Detection Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=data:image/png;base64," alt="YOLOv8">
  <img src="https://img.shields.io/badge/Gemini%20AI-Flash%202.0-orange?logo=google&logoColor=white" alt="Gemini AI">
  <img src="https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

> **Upload a leaf image → Get instant disease diagnosis + AI-powered treatment advice in seconds.**

CropGuard AI is an end-to-end, AI-powered crop disease detection web application. It combines a custom-trained **YOLOv8** object detection model with **Google Gemini 2.0 Flash Lite** to deliver structured agronomic advice — including organic/chemical treatments, prevention tips, severity assessment, and economic impact — directly from a leaf photograph.

---

## 📸 Features

| Feature | Details |
|---|---|
| 🔬 **YOLOv8 Detection** | Custom model trained on PlantVillage dataset (Tomato, Potato, Pepper & more) |
| 🤖 **Gemini AI Analysis** | Rich, structured JSON advice via Gemini 2.0 Flash Lite |
| 🧠 **Offline Knowledge Base** | Local fallback KB — works without internet / API key |
| 📋 **Severity & Urgency** | Low / Moderate / High / Critical + urgency timeline |
| 💊 **Dual Treatment Plans** | Both organic and chemical treatment options |
| 🌱 **Prevention Tips** | Actionable, crop-specific preventive advice |
| 📊 **Detection History** | SQLite-backed scan history with per-crop statistics |
| 🌐 **Modern Web UI** | Dark-mode, glassmorphism design with drag-and-drop upload |
| ⚡ **Quota-Aware** | Gracefully degrades to local KB on API quota exhaustion |

---

## 🏗️ Architecture

```
crop_disease_detection/
├── app/
│   ├── app.py              # Flask REST API + Web UI server
│   ├── database.py         # SQLite detection history & stats
│   ├── templates/
│   │   └── index.html      # Single-page application UI
│   └── static/
│       ├── style.css       # Dark-mode glassmorphism CSS
│       └── main.js         # Frontend logic (fetch, render, drag-drop)
│
├── src/
│   ├── ai_analyzer.py      # Gemini AI integration + local KB fallback
│   ├── predict.py          # YOLOv8 inference pipeline
│   ├── train_yolo.py       # YOLO training script
│   ├── train_densenet.py   # DenseNet-121 training script
│   ├── download_multicrop.py  # PlantVillage YOLO dataset builder
│   ├── download_beans.py   # Beans dataset downloader
│   ├── preprocess.py       # Image preprocessing utilities
│   └── evaluate.py         # Model evaluation metrics
│
├── models/                 # Trained model weights (not in git — see below)
│   ├── yolov8_custom.pt    # Custom-trained YOLOv8 weights
│   └── disease_info.json   # Disease metadata for 30+ classes
│
├── notebooks/              # Jupyter notebooks for EDA & experiments
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/HiteshPatil2610/CROP_DISEASE_DETECTION.git
cd CROP_DISEASE_DETECTION
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

Get a **free** Gemini API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)  
*(Free tier: 15 requests/min, 1M tokens/day — no credit card required)*

> **Note:** The app works fully without a Gemini API key. It falls back to the built-in disease knowledge base automatically.

### 5. Add your trained model

Place your trained YOLOv8 weights at:
```
models/yolov8_custom.pt
```

See [Training](#-training-your-own-model) below if you need to train one.

### 6. Run the application

```bash
python app/app.py
```

Open your browser at **http://localhost:5000** 🎉

---

## 🔌 REST API

The Flask backend exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/predict` | Upload leaf image → get full diagnosis |
| `POST` | `/api/ai-analysis` | On-demand Gemini re-analysis for crop + disease |
| `GET` | `/api/history` | Paginated detection history |
| `GET` | `/api/stats` | Aggregate statistics by crop and disease |
| `DELETE` | `/api/delete/<id>` | Delete a detection record |

### Example — `/api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@/path/to/leaf.jpg"
```

**Response:**
```json
{
  "crop_type": "Tomato",
  "disease_name": "Tomato — Early blight",
  "confidence": 87.5,
  "severity": "Moderate",
  "yolo_boxes": 2,
  "inference_ms": 312.4,
  "ai_analysis": {
    "condition_summary": "Early blight caused by Alternaria solani...",
    "severity_level": "Moderate",
    "key_symptoms": ["Dark concentric ring spots on lower leaves", "..."],
    "organic_treatments": ["Step 1: Remove infected leaves...", "..."],
    "chemical_treatments": ["Chlorothalonil (Bravo 720) — 2 ml/L..."],
    "prevention_tips": ["Rotate crops annually...", "..."],
    "urgency": "Within 48h",
    "economic_impact": "Can cause 30–50% yield loss if untreated.",
    "recovery_time": "2–3 weeks with consistent treatment.",
    "_source": "gemini"
  }
}
```

### Example — `/api/ai-analysis`

```bash
curl -X POST http://localhost:5000/api/ai-analysis \
  -H "Content-Type: application/json" \
  -d '{"crop": "Potato", "disease": "Late_blight", "confidence": 92.3}'
```

---

## 🌱 Supported Crops & Diseases

The model is trained on a subset of the **PlantVillage** dataset and supports:

| Crop | Diseases Detected |
|---|---|
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |

The local knowledge base additionally covers:
- Powdery Mildew, Leaf Rust, Bean Rust, Angular Leaf Spot, Mosaic Virus (generic)

---

## 🧠 AI Analysis Engine

### Gemini Integration (`src/ai_analyzer.py`)

When a Gemini API key is configured, the system calls **Gemini 2.0 Flash Lite** to generate structured JSON diagnostic advice:

```
Crop + Disease + Confidence  →  Gemini 2.0 Flash Lite  →  Structured JSON
```

Features:
- **Response caching** — LRU in-memory cache (up to 50 entries) to avoid repeat API calls
- **Quota handling** — Auto-detects `429 RESOURCE_EXHAUSTED` and switches to local KB for the session
- **Retry logic** — 2 automatic retries on transient errors with back-off
- **Strict JSON output** — Uses `response_mime_type="application/json"` for reliable parsing

### Local Knowledge Base (offline fallback)

When Gemini is unavailable, the app uses a hand-crafted knowledge base covering the most common diseases:

```
No API key / Quota exhausted  →  Disease KB lookup  →  Same JSON structure
```

The KB uses fuzzy matching (substring + word overlap) so it handles variant spellings robustly. If no match is found, it returns a generic-but-useful fallback rather than empty fields.

---

## 🎯 Training Your Own Model

### 1. Prepare dataset

```bash
# Download and structure PlantVillage dataset, then run:
python src/download_multicrop.py
```

This creates:
- `dataset/processed_multicrop/` — YOLO-formatted images + labels
- `dataset/multicrop_dataset.yaml` — YOLO training config
- `models/disease_info.json` — Disease metadata for 30+ classes

### 2. Train YOLOv8

```bash
python src/train_yolo.py
```

Or manually:
```bash
yolo detect train \
  data=dataset/multicrop_dataset.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=224 \
  batch=16 \
  name=cropguard_v1
```

Trained weights will be saved to `runs/detect/cropguard_v1/weights/best.pt`.  
Copy to `models/yolov8_custom.pt` when done.

### 3. Evaluate

```bash
python src/evaluate.py
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model inference & training |
| `flask` + `flask-cors` | Web server & REST API |
| `google-genai` | Gemini AI SDK |
| `opencv-python` | Image loading & annotation |
| `Pillow` | Image preprocessing |
| `python-dotenv` | Environment variable loading |
| `sqlite3` *(built-in)* | Detection history database |

Install all with:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key — enables AI analysis |

Copy `.env.example` → `.env` and fill in your key.

---

## 📁 Model Files (Not in Git)

Large binary files are excluded from version control via `.gitignore`. You must provide these yourself:

| File | Source |
|---|---|
| `models/yolov8_custom.pt` | Train with `src/train_yolo.py` or download from releases |
| `yolov8n.pt` / `yolov8m.pt` | [Ultralytics releases](https://github.com/ultralytics/assets/releases) |
| `dataset/` | [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — Hughes & Salathé (2015)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Google Gemini AI](https://ai.google.dev/)
- Inspired by precision agriculture research for smallholder farmers

---

<p align="center">Made with ❤️ for smarter, healthier crops 🌾</p>
