import os, sys, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')
from src.ai_analyzer import analyze_disease

print("Testing Tomato Early Blight...")
r = analyze_disease("Tomato", "Early Blight", 82.5)
print(f"Source: {r.get('_source')}")
print(f"Severity: {r.get('severity_level')}")
print(f"Organic treatments: {r.get('organic_treatments', [])[:2]}")
print(f"Prevention: {r.get('prevention_tips', [])[:2]}")

print("\nTesting Potato Late Blight...")
r2 = analyze_disease("Potato", "Late Blight", 74.0)
print(f"Source: {r2.get('_source')}")
print(f"Organic treatments: {r2.get('organic_treatments', [])[:2]}")
