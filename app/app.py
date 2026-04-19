"""
Flask Web Application
REST API + Web UI for crop disease detection
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path
import os, sys, json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import save_detection, get_all_detections, get_stats, delete_detection

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("app/static/uploads")
RESULT_FOLDER = Path("app/static/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ── Routes ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG, PNG, or WEBP"}), 400

    # Save uploaded image
    filename = f"upload_{os.urandom(4).hex()}_{file.filename}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        # Lazy import to avoid loading model on startup if not needed
        from src.predict import predict
        result = predict(str(save_path), save_result=True)
        detection_id = save_detection(result)
        result["detection_id"] = detection_id
        result["uploaded_image"] = f"/static/uploads/{filename}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def history():
    limit = int(request.args.get("limit", 50))
    rows = get_all_detections(limit)
    return jsonify(rows)

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(get_stats())

@app.route("/api/delete/<int:detection_id>", methods=["DELETE"])
def delete(detection_id):
    delete_detection(detection_id)
    return jsonify({"success": True, "deleted_id": detection_id})

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    print("🌿 CropGuard AI Server starting...")
    print("   Open: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)