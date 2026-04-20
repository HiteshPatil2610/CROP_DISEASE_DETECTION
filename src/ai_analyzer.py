"""
AI Disease Analyzer — Gemini 1.5 Flash Integration
Converts YOLOv8 detection output into rich, structured agronomic advice.

Usage:
    from src.ai_analyzer import analyze_disease
    result = analyze_disease("Tomato", "Early_blight", 87.5)
"""

import os
import json
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # dotenv not installed — rely on system env vars

# ── Gemini SDK lazy-import ──────────────────────────────────────────────────
_GEMINI_CLIENT    = None
_GEMINI_LOCK      = threading.Lock()
_GEMINI_AVAILABLE = None   # None = not yet tested
_QUOTA_EXHAUSTED  = False  # Set True on first 429 to skip API for rest of session

def _get_gemini_model():
    """Lazy-init Gemini. Returns model or None if unavailable."""
    global _GEMINI_CLIENT, _GEMINI_AVAILABLE
    if _GEMINI_AVAILABLE is False:
        return None
    with _GEMINI_LOCK:
        if _GEMINI_CLIENT is not None:
            return _GEMINI_CLIENT
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("[WARN] GEMINI_API_KEY not set -- AI analysis disabled. "
                  "Add it to your .env file to enable.")
            _GEMINI_AVAILABLE = False
            return None
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            _GEMINI_CLIENT = client
            _GEMINI_AVAILABLE = True
            print("[OK] Gemini 1.5 Flash AI analyzer ready.")
            return _GEMINI_CLIENT
        except Exception as exc:
            print(f"[WARN] Gemini init failed: {exc}")
            _GEMINI_AVAILABLE = False
            return None


# ── Prompt Template ──────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """You are an expert agricultural pathologist and agronomist with 20 years of field experience.

A YOLOv8 deep-learning model detected the following on a crop leaf image:
- Crop Type   : {crop}
- Disease     : {disease}
- Confidence  : {confidence:.1f}%

Provide a comprehensive, structured diagnostic response in this exact JSON format:
{{
  "condition_summary": "2-3 sentence plain-language overview of this specific disease, its cause (fungal/bacterial/viral/pest), and how it spreads.",
  "severity_level": "Low | Moderate | High | Critical",
  "key_symptoms": ["visible symptom 1", "visible symptom 2", "visible symptom 3"],
  "organic_treatments": [
    "Step 1: specific organic action with product name if applicable",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "chemical_treatments": [
    "Product name + active ingredient + dosage + application method",
    "Alternative product if primary is unavailable"
  ],
  "prevention_tips": [
    "Actionable prevention tip 1",
    "Actionable prevention tip 2",
    "Actionable prevention tip 3"
  ],
  "urgency": "Immediate | Within 48h | Within 1 week | Monitor only",
  "economic_impact": "One sentence on potential yield loss if untreated.",
  "recovery_time": "Estimated crop recovery period after treatment."
}}

Rules:
- Return ONLY valid JSON — no markdown, no extra text.
- Be specific to {crop} crops and {disease} disease.
- Use farmer-friendly language in descriptions; be precise with product names.
- If the disease appears healthy or benign, set severity_level to "None" and urgency to "Monitor only".
"""


# ── In-memory LRU cache (keyed on normalized disease label) ─────────────────
_ANALYSIS_CACHE: dict[str, dict] = {}
_CACHE_MAX      = 50   # Maximum cached entries

def _cache_get(key: str) -> Optional[dict]:
    return _ANALYSIS_CACHE.get(key)

def _cache_set(key: str, value: dict):
    if len(_ANALYSIS_CACHE) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(_ANALYSIS_CACHE))
        del _ANALYSIS_CACHE[oldest]
    _ANALYSIS_CACHE[key] = value


# ── Disease Knowledge Base (local fallback when Gemini is unavailable) ────────
_DISEASE_KB: dict = {
    # ── Fungal diseases ────────────────────────────────────────────────────────
    "early blight": {
        "severity_level": "Moderate",
        "key_symptoms": ["Dark concentric ring spots on lower leaves", "Yellow halo around lesions", "Leaf yellowing and drop"],
        "organic_treatments": [
            "Step 1: Remove and destroy all infected leaves immediately.",
            "Step 2: Apply copper-based fungicide spray (copper hydroxide, 3g/L water) every 7 days.",
            "Step 3: Spray neem oil solution (5 ml/L water + 2 ml dish soap) on remaining foliage.",
        ],
        "chemical_treatments": [
            "Chlorothalonil (Bravo 720) — 2 ml/L, spray every 7–10 days.",
            "Mancozeb 75 WP — 2.5 g/L water, apply at first sign of disease.",
        ],
        "prevention_tips": [
            "Rotate crops — avoid planting the same family in the same bed for 2–3 years.",
            "Water at the base; avoid overhead irrigation to keep foliage dry.",
            "Stake plants for better airflow and reduce humidity around leaves.",
        ],
        "urgency": "Within 48h",
        "economic_impact": "Can cause 30–50% yield loss if untreated during humid seasons.",
        "recovery_time": "2–3 weeks with consistent treatment.",
    },
    "late blight": {
        "severity_level": "Critical",
        "key_symptoms": ["Water-soaked lesions on leaves and stems", "White mouldy growth on leaf undersides", "Rapid browning and collapse of tissue"],
        "organic_treatments": [
            "Step 1: Remove and bag all infected material — do NOT compost.",
            "Step 2: Apply copper oxychloride (50 WP, 3 g/L) every 5–7 days.",
            "Step 3: Improve field drainage; avoid waterlogging.",
        ],
        "chemical_treatments": [
            "Metalaxyl + Mancozeb (Ridomil Gold) — 2.5 g/L, apply every 7 days.",
            "Cymoxanil + Mancozeb — 2 g/L water as preventive spray.",
        ],
        "prevention_tips": [
            "Plant certified disease-free seed/seedlings.",
            "Apply preventive copper sprays before wet weather forecasts.",
            "Maintain wide plant spacing (60 cm+) to reduce humidity buildup.",
        ],
        "urgency": "Immediate",
        "economic_impact": "Can cause 100% crop loss in 7–10 days under favourable weather if untreated.",
        "recovery_time": "4–6 weeks with aggressive treatment; severe cases may not recover.",
    },
    "leaf rust": {
        "severity_level": "Moderate",
        "key_symptoms": ["Orange-brown powdery pustules on leaf surface", "Pustules on both sides of leaf", "Yellowing of surrounding tissue"],
        "organic_treatments": [
            "Step 1: Prune and destroy affected leaves.",
            "Step 2: Apply sulfur-based fungicide (wettable sulfur 80%, 2 g/L) every 10 days.",
            "Step 3: Spray garlic extract solution (50 g crushed garlic in 1L water, dilute 1:10).",
        ],
        "chemical_treatments": [
            "Tebuconazole (Folicur 250 EW) — 1 ml/L, spray at first signs.",
            "Propiconazole (Tilt 250 EC) — 1 ml/L water every 14 days.",
        ],
        "prevention_tips": [
            "Plant rust-resistant varieties where available.",
            "Avoid dense planting; prune for good air circulation.",
            "Remove volunteer plants and weed hosts around the field.",
        ],
        "urgency": "Within 48h",
        "economic_impact": "Can reduce yield by 20–40% if infection spreads during grain filling.",
        "recovery_time": "3–4 weeks with appropriate fungicide treatment.",
    },
    "powdery mildew": {
        "severity_level": "Low",
        "key_symptoms": ["White powdery coating on leaf surfaces", "Curling and distortion of young leaves", "Stunted growth of new shoots"],
        "organic_treatments": [
            "Step 1: Mix 1 tsp baking soda + 1 tsp neem oil + 1L water; spray weekly.",
            "Step 2: Apply potassium bicarbonate (10 g/L) as a contact fungicide.",
            "Step 3: Remove heavily infected shoots to reduce spore load.",
        ],
        "chemical_treatments": [
            "Myclobutanil (Eagle 20EW) — 1 ml/L, effective systemic fungicide.",
            "Sulfur dust or wettable sulfur — apply in the evening to avoid phytotoxicity.",
        ],
        "prevention_tips": [
            "Avoid over-fertilizing with nitrogen — promotes susceptible soft growth.",
            "Water early in the morning so foliage dries before evening.",
            "Maintain 30–40% relative humidity if growing in enclosed spaces.",
        ],
        "urgency": "Within 1 week",
        "economic_impact": "Mainly aesthetic; can reduce photosynthesis and fruit quality by 10–20%.",
        "recovery_time": "1–2 weeks after treatment begins.",
    },
    "angular leaf spot": {
        "severity_level": "Moderate",
        "key_symptoms": ["Angular water-soaked spots on leaves", "Spots turn brown with yellow margins", "Bacterial ooze visible in humid conditions"],
        "organic_treatments": [
            "Step 1: Remove and destroy infected plant debris.",
            "Step 2: Apply copper hydroxide spray (50 WP, 3 g/L) at 7-day intervals.",
            "Step 3: Avoid working in fields when plants are wet.",
        ],
        "chemical_treatments": [
            "Copper oxychloride (50 WP) — 3 g/L water, spray every 7–10 days.",
            "Streptomycin sulfate (100 ppm) — apply at disease onset for bacterial control.",
        ],
        "prevention_tips": [
            "Use certified disease-free seed.",
            "Practice 2-year crop rotation away from beans/legumes.",
            "Avoid overhead irrigation; water at the base.",
        ],
        "urgency": "Within 48h",
        "economic_impact": "Can cause 30–60% yield loss in severe outbreaks.",
        "recovery_time": "3–4 weeks with consistent copper treatment.",
    },
    "bean rust": {
        "severity_level": "Moderate",
        "key_symptoms": ["Brick-red to brown pustules on leaf undersides", "Yellow spots on upper leaf surface", "Premature leaf drop"],
        "organic_treatments": [
            "Step 1: Remove heavily infected leaves and dispose of away from field.",
            "Step 2: Spray wettable sulfur (2 g/L) every 10 days.",
            "Step 3: Apply compost tea as a foliar spray to boost natural resistance.",
        ],
        "chemical_treatments": [
            "Tebuconazole (Folicur) — 1 ml/L water, systemic protection.",
            "Chlorothalonil (Bravo) — 2 ml/L, protective cover spray.",
        ],
        "prevention_tips": [
            "Plant rust-resistant bean varieties (e.g., NABE 15, K132).",
            "Increase plant spacing to reduce humidity.",
            "Rotate beans with non-legume crops for at least 2 seasons.",
        ],
        "urgency": "Within 48h",
        "economic_impact": "Can reduce yield by 20–50% in susceptible varieties.",
        "recovery_time": "2–4 weeks with fungicide application.",
    },
    "mosaic virus": {
        "severity_level": "High",
        "key_symptoms": ["Mottled yellow-green mosaic pattern on leaves", "Leaf crinkle and distortion", "Stunted plant growth"],
        "organic_treatments": [
            "Step 1: Remove and destroy infected plants — there is no cure for viral infection.",
            "Step 2: Control aphid vectors with insecticidal soap spray (5 ml/L water).",
            "Step 3: Use reflective mulch to deter aphid vectors.",
        ],
        "chemical_treatments": [
            "Imidacloprid (Confidor) — 0.5 ml/L, systemic aphid control to reduce virus spread.",
            "Mineral oil spray — can reduce aphid transmission by 50% if applied early.",
        ],
        "prevention_tips": [
            "Use virus-free certified seed and resistant varieties.",
            "Control aphid populations with regular monitoring and early insecticide application.",
            "Remove and destroy infected plants immediately — viruses spread rapidly.",
        ],
        "urgency": "Immediate",
        "economic_impact": "Infected plants rarely recover; 40–80% yield loss depending on infection timing.",
        "recovery_time": "No recovery — remove infected plants and replant with resistant varieties.",
    },
}

def _get_kb_entry(disease: str) -> Optional[dict]:
    """Fuzzy-match disease name to knowledge base entry."""
    d_lower = disease.lower().replace("_", " ").replace("-", " ")
    for key, data in _DISEASE_KB.items():
        if key in d_lower or d_lower in key:
            return data
    # Check individual words
    d_words = set(d_lower.split())
    for key, data in _DISEASE_KB.items():
        k_words = set(key.split())
        if len(d_words & k_words) >= 2:  # At least 2 matching words
            return data
    return None


def _build_fallback(crop: str, disease: str, confidence: float) -> dict:
    """
    Returns a disease-specific local knowledge base response when Gemini is unavailable.
    Falls back to generic advice only if the disease is truly unknown.
    """
    is_healthy = "healthy" in disease.lower()
    if is_healthy:
        return {
            "condition_summary": f"{crop} plant appears healthy — no disease detected.",
            "severity_level"  : "None",
            "key_symptoms"    : [],
            "organic_treatments": ["Continue regular monitoring.", "Maintain balanced nutrition and watering schedule."],
            "chemical_treatments": [],
            "prevention_tips": [
                "Monitor plants weekly for early signs of disease.",
                "Maintain proper plant spacing for air circulation.",
                "Apply preventive foliar nutrition at recommended intervals.",
            ],
            "urgency"         : "Monitor only",
            "economic_impact" : "N/A — plant is healthy.",
            "recovery_time"   : "N/A",
            "_source"         : "local-kb",
        }

    # Try disease-specific knowledge base
    kb = _get_kb_entry(disease)
    if kb:
        return {
            "condition_summary": f"{disease.replace('_', ' ').title()} detected on {crop} plant. {kb.get('key_symptoms', [''])[0] if kb.get('key_symptoms') else ''}",
            **kb,
            "_source": "local-kb",
        }

    # True generic fallback for unknown diseases
    return {
        "condition_summary": (
            f"{disease.replace('_', ' ').title()} detected on {crop} plant. "
            "Isolate affected plants and consult a local agronomist for targeted treatment."
        ),
        "severity_level"  : "Unknown",
        "key_symptoms"    : ["Visible lesions or discoloration", "Abnormal growth patterns", "Possible wilting"],
        "organic_treatments": [
            "Step 1: Remove and destroy all visibly infected plant material.",
            "Step 2: Apply neem oil spray (5 ml/L water + few drops dish soap) every 7 days.",
            "Step 3: Improve drainage and air circulation around plants.",
        ],
        "chemical_treatments": [
            "Consult your local agronomist for a precise chemical treatment plan.",
            "Broad-spectrum fungicide (e.g., Mancozeb 75 WP, 2.5 g/L) as interim measure.",
        ],
        "prevention_tips": [
            f"Practice 2–3 year crop rotation away from {crop}.",
            "Avoid overhead irrigation — water at the base to keep foliage dry.",
            "Inspect and remove volunteer plants and weeds that may harbour disease.",
        ],
        "urgency"         : "Within 1 week",
        "economic_impact" : f"Potential yield reduction if {disease.replace('_', ' ')} spreads unchecked.",
        "recovery_time"   : "Varies — typically 2–4 weeks with appropriate treatment.",
        "_source"         : "fallback",
    }


# ── Main public function ─────────────────────────────────────────────────────
def analyze_disease(crop: str, disease: str, confidence: float) -> dict:
    """
    Analyze a detected crop disease using Gemini 1.5 Flash.

    Args:
        crop       : Crop type (e.g. "Tomato", "Potato")
        disease    : Disease label from YOLO (e.g. "Early_blight")
        confidence : YOLO confidence score (0-100)

    Returns:
        dict with keys: condition_summary, severity_level, key_symptoms,
        organic_treatments, chemical_treatments, prevention_tips,
        urgency, economic_impact, recovery_time, _source
    """
    global _QUOTA_EXHAUSTED
    cache_key = f"{crop.lower()}::{disease.lower()}"

    # Return cached result if available
    cached = _cache_get(cache_key)
    if cached:
        return {**cached, "_source": "cache"}

    model = _get_gemini_model()
    if model is None or _QUOTA_EXHAUSTED:
        return _build_fallback(crop, disease, confidence)

    prompt = ANALYSIS_PROMPT.format(
        crop=crop,
        disease=disease.replace("_", " ").replace("___", " -- "),
        confidence=confidence,
    )

    # Retry up to 2 times on transient errors
    for attempt in range(2):
        try:
            from google import genai
            from google.genai import types
            t0 = time.time()
            response = model.models.generate_content(
                model="models/gemini-2.0-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            elapsed  = (time.time() - t0) * 1000

            raw_text = response.text.strip()

            # Strip accidental markdown fences if the model adds them
            if raw_text.startswith("```"):
                raw_text = "\n".join(raw_text.split("\n")[1:])
                raw_text = raw_text.rstrip("` \n")

            parsed = json.loads(raw_text)
            parsed["_source"]      = "gemini"
            parsed["_response_ms"] = round(elapsed, 1)

            _cache_set(cache_key, parsed)
            return parsed

        except json.JSONDecodeError as e:
            print(f"[WARN] Gemini JSON parse error (attempt {attempt+1}): {e}")
            if attempt == 1:
                return _build_fallback(crop, disease, confidence)
            time.sleep(0.5)

        except Exception as e:
            err_str = str(e)
            print(f"[WARN] Gemini API error (attempt {attempt+1}): {err_str[:200]}")
            # Rate limit — set session flag and return local KB immediately
            if "429" in err_str or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                _QUOTA_EXHAUSTED = True
                print("[WARN] Quota exhausted -- switching to local knowledge base for this session.")
                return _build_fallback(crop, disease, confidence)
            if attempt == 1:
                return _build_fallback(crop, disease, confidence)
            time.sleep(1)

    return _build_fallback(crop, disease, confidence)


# ── CLI quick-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    crop    = sys.argv[1] if len(sys.argv) > 1 else "Tomato"
    disease = sys.argv[2] if len(sys.argv) > 2 else "Early_blight"
    conf    = float(sys.argv[3]) if len(sys.argv) > 3 else 87.5

    print(f"\n🔬 Analyzing: {crop} — {disease} ({conf}%)\n")
    result = analyze_disease(crop, disease, conf)
    print(json.dumps(result, indent=2))
