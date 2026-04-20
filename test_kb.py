from src.ai_analyzer import analyze_disease

tests = [
    ("Tomato", "Early blight", 82.5),
    ("Bean", "Angular Leaf Spot", 76.3),
    ("Tomato", "Late blight", 91.0),
    ("Wheat", "Leaf Rust", 68.0),
    ("Unknown", "SomeNewDisease", 55.0),
    ("Bean", "Bean Healthy", 88.0),
]

for crop, disease, conf in tests:
    r = analyze_disease(crop, disease, conf)
    src = r["_source"]
    sev = r["severity_level"]
    org = r["organic_treatments"][0][:55]
    tip = r["prevention_tips"][0][:55]
    print(f"{crop}/{disease}")
    print(f"  source={src}, severity={sev}")
    print(f"  organic: {org}")
    print(f"  tip:     {tip}")
    print()
