# ============================================================
#   BANANA LEAF DISEASE — FLASK API SERVER
#   Loads trained model into memory on startup
#   Serves predictions via HTTP endpoints
# ============================================================

import os
import json
import io
import numpy as np
import colorsys
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image

# ── Initialize Flask App ─────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')

# ── Global Variables (Model in Memory) ───────────────────────
MODEL = None
CLASS_NAMES = None
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
GREEN_RATIO_THRESHOLD = 0.10

DISEASE_INFO = {
    "healthy": {
        "emoji": "✅",
        "title": "HEALTHY LEAF",
        "description": "Your banana leaf is perfectly healthy. No disease detected.",
        "solutions": [
            "Keep watering the plant regularly",
            "Ensure it receives adequate sunlight",
            "Remove dead or yellowing leaves periodically",
            "Monitor the plant weekly for early signs of disease"
        ]
    },
    "cordana": {
        "emoji": "🟡",
        "title": "CORDANA LEAF SPOT",
        "description": "A fungal disease caused by Cordana musae. Appears as oval brown spots with yellow borders.",
        "solutions": [
            "Remove and destroy all infected leaves immediately",
            "Spray a copper-based fungicide on the plant",
            "Avoid wetting the leaves while watering — water the soil only",
            "Improve air flow by spacing plants adequately",
            "Apply fungicide again after 10–14 days if needed"
        ]
    },
    "pestalotiopsis": {
        "emoji": "🟠",
        "title": "PESTALOTIOPSIS",
        "description": "A fungal infection that thrives in wet and humid conditions. Causes dark lesions on leaf surface.",
        "solutions": [
            "Cut off and burn all visibly infected leaves",
            "Apply a broad-spectrum fungicide such as Mancozeb or Chlorothalonil",
            "Avoid waterlogging — ensure proper soil drainage",
            "Keep the field clean of fallen or decaying leaves",
            "Reduce humidity around plants where possible"
        ]
    },
    "sigatoka": {
        "emoji": "🔴",
        "title": "SIGATOKA (Black/Yellow Sigatoka)",
        "description": "One of the most damaging banana diseases worldwide. Caused by fungal pathogens, it severely reduces yield.",
        "solutions": [
            "Remove and safely dispose of heavily infected leaves right away",
            "Spray systemic fungicides like Propiconazole or Tridemorph",
            "Rotate between different fungicide types to prevent resistance",
            "Ensure good field drainage to reduce leaf wetness",
            "Schedule regular preventive spraying during wet seasons"
        ]
    }
}

# ── Helper Functions ─────────────────────────────────────────
def is_green_enough(pil_img, threshold=GREEN_RATIO_THRESHOLD):
    """Check if image has enough green pixels (is likely a leaf)"""
    small = pil_img.resize((100, 100)).convert("RGB")
    pixels = np.array(small).reshape(-1, 3)
    
    green_count = 0
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        if 0.15 <= h <= 0.55 and s > 0.15 and v > 0.15:
            green_count += 1
    
    ratio = green_count / len(pixels)
    green_ok = ratio >= threshold
    return green_ok, ratio

def is_confident_enough(preds, threshold=CONFIDENCE_THRESHOLD):
    """Check if model is confident in prediction"""
    top_conf = float(np.max(preds))
    conf_ok = top_conf >= threshold
    return conf_ok, top_conf

# ── Model Loading ────────────────────────────────────────────
def ensure_model_loaded():
    """Load model into memory once when an API endpoint needs it."""
    global MODEL, CLASS_NAMES

    if MODEL is not None and CLASS_NAMES is not None:
        return None

    print("\n🔄 Loading model into memory...")
    try:
        MODEL = tf.keras.models.load_model("models/banana_disease_model.h5")
        print("✅ Model loaded successfully into memory")

        with open("models/class_metadata.json", 'r') as f:
            metadata = json.load(f)
            CLASS_NAMES = metadata["class_names"]
        print(f"✅ Classes loaded: {CLASS_NAMES}")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return str(e)

# ── API Routes ───────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    """Serve the web interface"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if MODEL is None or CLASS_NAMES is None:
        return jsonify({"status": "loading model"}), 202
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "classes_available": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    })

@app.route('/info', methods=['GET'])
def info():
    """Get model and disease information"""
    load_error = ensure_model_loaded()
    if load_error:
        return jsonify({"error": f"Model loading failed: {load_error}"}), 500

    return jsonify({
        "model_name": "ResNet50 Banana Leaf Disease Classifier",
        "classes": CLASS_NAMES,
        "expected_input": "Image file (JPG, PNG) of banana leaf",
        "validation_thresholds": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "green_ratio_threshold": GREEN_RATIO_THRESHOLD
        },
        "diseases": {
            name: {
                "emoji": info.get("emoji"),
                "title": info.get("title"),
                "description": info.get("description")
            }
            for name, info in DISEASE_INFO.items()
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict banana leaf disease from image
    
    Expected: multipart/form-data with 'image' file
    Returns: JSON with prediction, confidence, and treatment info
    """
    
    load_error = ensure_model_loaded()
    if load_error:
        return jsonify({"error": f"Model loading failed: {load_error}"}), 500
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No filename provided"}), 400
    
    try:
        # Load image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        
        # ── Step 1: Green pixel ratio check ──────────────────
        green_ok, green_ratio = is_green_enough(img)
        
        if not green_ok:
            return jsonify({
                "status": "rejected",
                "reason": "NOT_A_LEAF",
                "message": "Image does not contain enough green content to be a banana leaf",
                "green_ratio": float(green_ratio),
                "required_ratio": GREEN_RATIO_THRESHOLD
            }), 400
        
        # ── Step 2: Prepare image for model ──────────────────
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # ── Step 3: Make prediction ──────────────────────────
        preds = MODEL.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])
        
        # ── Step 4: Confidence gate ──────────────────────────
        conf_ok, top_conf = is_confident_enough(preds)
        
        if not conf_ok:
            return jsonify({
                "status": "rejected",
                "reason": "LOW_CONFIDENCE",
                "message": "Model could not confidently classify this image",
                "confidence": confidence,
                "required_confidence": CONFIDENCE_THRESHOLD
            }), 400
        
        # ── Step 5: Return successful prediction ──────────────
        class_probabilities = {
            CLASS_NAMES[i]: float(preds[i])
            for i in range(len(CLASS_NAMES))
        }
        
        disease_info = DISEASE_INFO.get(pred_class, {})
        
        return jsonify({
            "status": "success",
            "predicted_class": pred_class,
            "confidence": confidence,
            "all_probabilities": class_probabilities,
            "validation": {
                "green_ratio": float(green_ratio),
                "passes_green_check": True
            },
            "disease_info": {
                "emoji": disease_info.get("emoji"),
                "title": disease_info.get("title"),
                "description": disease_info.get("description"),
                "recommended_solutions": disease_info.get("solutions", [])
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# ── Error Handlers ───────────────────────────────────────────
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 70)
    print("🚀 BANANA LEAF DISEASE CLASSIFIER - API SERVER")
    print("=" * 70)
    print(f"Starting Flask server on port {port}...")
    print("Model will be loaded on first request and kept in memory\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
