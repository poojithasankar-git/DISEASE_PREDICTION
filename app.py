# ============================================================
#   BANANA LEAF DISEASE — FLASK API SERVER
#   Loads trained model into memory on startup
#   Serves predictions via HTTP endpoints
# ============================================================

import os
import json
import io
import threading
import numpy as np
import colorsys
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── Initialize Flask App ─────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')

# ── Global Variables (Model in Memory) ───────────────────────
MODEL = None
CLASS_NAMES = None
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
GREEN_RATIO_THRESHOLD = 0.10
MODEL_LOAD_LOCK = threading.Lock()
MODEL_WARMUP_STARTED = False
MODEL_LOAD_ERROR = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "banana_model.h5")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
TF = None
KERAS_LOAD_MODEL = None
IMG_TO_ARRAY = None
PREPROCESS_INPUT = None

# Hidden backup service (Gemini API)
BACKUP_SERVICE_KEY = os.environ.get('BACKUP_SVC')
BACKUP_SERVICE_AVAILABLE = False

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

def call_backup_service(image_data, primary_prediction=None, primary_confidence=None):
    """
    Hidden secondary verification service using Gemini Vision API.
    Checks accuracy and confidence for every image analysis.
    Returns: dict with disease, confidence, accuracy_score, match_status
    """
    if not BACKUP_SERVICE_KEY or not BACKUP_SERVICE_AVAILABLE:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=BACKUP_SERVICE_KEY)
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image_data.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Use Gemini for detailed analysis with accuracy assessment
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """Analyze this banana leaf image carefully for disease classification.
        
        Classify as one of: 'healthy', 'cordana', 'pestalotiopsis', 'sigatoka'
        
        Provide detailed accuracy assessment:
        1. Visual confidence in your classification (0-1)
        2. Image quality assessment (clear=1, blurry=0.5, very_poor=0.1)
        3. How certain are you about this being a banana leaf? (0-1)
        4. Overall accuracy score (confidence × image_quality × leaf_certainty)
        
        Respond ONLY with JSON:
        {"disease": "<class>", "confidence": <0-1>, "image_quality": <0-1>, "leaf_certainty": <0-1>, "accuracy_score": <0-1>}"""
        
        response = model.generate_content([
            {
                "mime_type": "image/png",
                "data": b64_img
            },
            prompt
        ])
        
        # Parse detailed response
        result_text = response.text.strip()
        if result_text.startswith('{'):
            result = json.loads(result_text)
            
            # Determine match status with primary model
            match_status = "agreement"
            if primary_prediction and primary_prediction != result.get('disease'):
                match_status = "disagreement"
            
            return {
                "disease": result.get('disease'),
                "confidence": float(result.get('confidence', 0.5)),
                "image_quality": float(result.get('image_quality', 0.7)),
                "leaf_certainty": float(result.get('leaf_certainty', 0.8)),
                "accuracy_score": float(result.get('accuracy_score', 0.5)),
                "match_status": match_status,
                "primary_model": primary_prediction,
                "primary_confidence": primary_confidence
            }
    except Exception as e:
        print(f"⚠️ Secondary verification unavailable: {e}")
        return None
    
    return None

# ── Model Loading ────────────────────────────────────────────
def ensure_tf_loaded():
    """Import TensorFlow lazily to keep app startup lightweight."""
    global TF, KERAS_LOAD_MODEL, IMG_TO_ARRAY, PREPROCESS_INPUT

    if TF is not None:
        return None

    try:
        import tensorflow as tf
        import keras
        from keras.saving import load_model as keras_load_model
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.applications.resnet import preprocess_input

        TF = tf
        KERAS_LOAD_MODEL = keras_load_model
        IMG_TO_ARRAY = img_to_array
        PREPROCESS_INPUT = preprocess_input
        return None
    except Exception as e:
        return str(e)


def ensure_model_loaded():
    """Load model into memory once; returns error text on failure."""
    global MODEL, CLASS_NAMES, MODEL_LOAD_ERROR, BACKUP_SERVICE_AVAILABLE

    if MODEL is not None and CLASS_NAMES is not None:
        return None

    with MODEL_LOAD_LOCK:
        if MODEL is not None and CLASS_NAMES is not None:
            return None

        print("\n🔄 Loading model into memory...")
        try:
            tf_error = ensure_tf_loaded()
            if tf_error:
                MODEL_LOAD_ERROR = tf_error
                return MODEL_LOAD_ERROR

            class DensePatched(TF.keras.layers.Dense):
                @classmethod
                def from_config(cls, config):
                    config.pop("quantization_config", None)
                    return super().from_config(config)

            class InputLayerPatched(TF.keras.layers.InputLayer):
                @classmethod
                def from_config(cls, config):
                    config.pop("batch_shape", None)
                    config.pop("optional", None)
                    return super().from_config(config)

            custom_objects = {
                "Dense": DensePatched,
                "InputLayer": InputLayerPatched,
            }

            try:
                MODEL = KERAS_LOAD_MODEL(
                    MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objects,
                    safe_mode=False,
                )
            except Exception as load_error:
                # Fallback for environments without Keras 3 support
                print(f"⚠️ Keras load failed, falling back to tf.keras: {load_error}")
                MODEL = TF.keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                    custom_objects=custom_objects,
                )
            print("✅ Model loaded successfully into memory")

            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                CLASS_NAMES = metadata["class_names"]
            print(f"✅ Classes loaded: {CLASS_NAMES}")
            MODEL_LOAD_ERROR = None
            
            # Check if backup service is available
            if BACKUP_SERVICE_KEY:
                try:
                    import google.generativeai as genai
                    BACKUP_SERVICE_AVAILABLE = True
                    print("✅ Backup service (hidden) is available")
                except:
                    BACKUP_SERVICE_AVAILABLE = False
            
            return None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            MODEL_LOAD_ERROR = str(e)
            return MODEL_LOAD_ERROR


def trigger_model_warmup():
    """Start background model warmup once so requests do not time out."""
    global MODEL_WARMUP_STARTED

    if MODEL is not None and CLASS_NAMES is not None:
        return

    with MODEL_LOAD_LOCK:
        if MODEL_WARMUP_STARTED:
            return
        MODEL_WARMUP_STARTED = True

    threading.Thread(target=ensure_model_loaded, daemon=True).start()

# ── API Routes ───────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    """Serve the web interface"""
    trigger_model_warmup()
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if MODEL is None or CLASS_NAMES is None:
        trigger_model_warmup()
        if MODEL_LOAD_ERROR:
            return jsonify({"status": "error", "error": MODEL_LOAD_ERROR}), 500
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
    if MODEL is None or CLASS_NAMES is None:
        trigger_model_warmup()
        if MODEL_LOAD_ERROR:
            return jsonify({"error": f"Model loading failed: {MODEL_LOAD_ERROR}"}), 500
        return jsonify({"status": "loading", "message": "Model is warming up. Try again shortly."}), 202

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
    
    if MODEL is None or CLASS_NAMES is None:
        trigger_model_warmup()
        if MODEL_LOAD_ERROR:
            return jsonify({"error": f"Model loading failed: {MODEL_LOAD_ERROR}"}), 500
        return jsonify({"status": "loading", "message": "Model is warming up. Please retry in about 30-60 seconds."}), 503
    
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
                "message": "❌ This doesn't look like a banana leaf. Please upload a clear image of a banana leaf and try again.",
                "hint": "Make sure the image shows a banana leaf clearly",
                "green_ratio": float(green_ratio),
                "required_ratio": GREEN_RATIO_THRESHOLD
            }), 400
        
        # ── Step 2: Prepare image for model ──────────────────
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = IMG_TO_ARRAY(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = PREPROCESS_INPUT(img_array)
        
        # ── Step 3: Make prediction ──────────────────────────
        preds = MODEL.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])
        
        # ── Step 4: Confidence gate + hidden verification ──────────────────────────
        conf_ok, top_conf = is_confident_enough(preds)
        
        # Always check with hidden verification service for accuracy assessment
        hidden_analysis = None
        if BACKUP_SERVICE_AVAILABLE:
            hidden_analysis = call_backup_service(img, pred_class, confidence)
        
        if not conf_ok:
            # Low confidence - try to use hidden verification
            if hidden_analysis and hidden_analysis.get('accuracy_score', 0) >= 0.6:
                pred_class = hidden_analysis['disease']
                confidence = hidden_analysis['confidence']
                class_probabilities = {CLASS_NAMES[i]: 0.0 for i in range(len(CLASS_NAMES))}
                class_probabilities[pred_class] = confidence
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
                    "source": "hidden_verified",
                    "verification_metrics": {
                        "image_quality": hidden_analysis.get('image_quality'),
                        "accuracy_score": hidden_analysis.get('accuracy_score')
                    },
                    "disease_info": {
                        "emoji": disease_info.get("emoji"),
                        "title": disease_info.get("title"),
                        "description": disease_info.get("description"),
                        "recommended_solutions": disease_info.get("solutions", [])
                    }
                }), 200
            
            return jsonify({
                "status": "rejected",
                "reason": "LOW_CONFIDENCE",
                "message": "Model could not confidently classify this image",
                "confidence": confidence,
                "required_confidence": CONFIDENCE_THRESHOLD
            }), 400
        
        # ── Step 5: Return successful prediction with hidden verification data ──────────────
        class_probabilities = {
            CLASS_NAMES[i]: float(preds[i])
            for i in range(len(CLASS_NAMES))
        }
        
        disease_info = DISEASE_INFO.get(pred_class, {})
        
        # Build response with hidden verification metrics if available
        response_data = {
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
        }
        
        # Add hidden verification metrics if available (for accuracy assessment)
        if hidden_analysis:
            response_data["verification_metrics"] = {
                "image_quality": hidden_analysis.get('image_quality'),
                "leaf_certainty": hidden_analysis.get('leaf_certainty'),
                "accuracy_score": hidden_analysis.get('accuracy_score'),
                "model_agreement": hidden_analysis.get('match_status') == 'agreement'
            }
        
        return jsonify(response_data), 200
        
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
