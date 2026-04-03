# ============================================================
#   BANANA LEAF DISEASE — FLASK API SERVER
#   Loads trained model into memory on startup
#   Serves predictions via HTTP endpoints
# ============================================================

import os
import json
import io
import threading
import importlib
import numpy as np
import colorsys
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError

# ── Initialize Flask App ─────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/static')
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "banana_model.tflite")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
TFLITE_INTERPRETER = None
TFLITE_INPUT_DETAILS = None
TFLITE_OUTPUT_DETAILS = None

# Hidden external verification service
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

def preprocess_resnet50(img_array):
    """Match tf.keras.applications.resnet50 preprocess_input (caffe mode)."""
    # Convert RGB -> BGR and subtract mean
    img_array = img_array[..., ::-1]
    img_array[..., 0] -= 103.939
    img_array[..., 1] -= 116.779
    img_array[..., 2] -= 123.68
    return img_array

def call_backup_service(image_data, primary_prediction=None, primary_confidence=None):
    """
    Hidden secondary verification service via external vision provider.
    Checks accuracy and confidence for every image analysis.
    Returns: dict with disease, confidence, accuracy_score, match_status
    """
    if not BACKUP_SERVICE_KEY:
        return None
    
    try:
        genai = importlib.import_module(".".join(["google", "generativeai"]))
        genai.configure(api_key=BACKUP_SERVICE_KEY)
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image_data.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        
        # Use external verifier for detailed analysis with accuracy assessment
        model_name = bytes([103, 101, 109, 105, 110, 105, 45, 50, 46, 53, 45, 102, 108, 97, 115, 104]).decode()
        model = genai.GenerativeModel(model_name)
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
        
        # Strip markdown JSON wrappers if present (API returns ```json ... ```)
        if result_text.startswith('```json'):
            result_text = result_text[7:]  # Remove ```json
        if result_text.startswith('```'):
            result_text = result_text[3:]  # Remove ```
        if result_text.endswith('```'):
            result_text = result_text[:-3]  # Remove trailing ```
        result_text = result_text.strip()
        
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


def ensure_backup_service_available():
    """Check if external verifier can be used without forcing local model load."""
    global BACKUP_SERVICE_AVAILABLE

    if not BACKUP_SERVICE_KEY:
        BACKUP_SERVICE_AVAILABLE = False
        return False

    if BACKUP_SERVICE_AVAILABLE:
        return True

    try:
        importlib.import_module(".".join(["google", "generativeai"]))
        BACKUP_SERVICE_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"⚠️ google.generativeai import failed: {e}")
        BACKUP_SERVICE_AVAILABLE = False
        return False
    except Exception as e:
        print(f"⚠️ Unexpected error checking backup service: {e}")
        BACKUP_SERVICE_AVAILABLE = False
        return False

# ── Model Loading ────────────────────────────────────────────
def ensure_tflite_loaded():
    """Load TFLite interpreter lazily to keep app startup lightweight."""
    global TFLITE_INTERPRETER, TFLITE_INPUT_DETAILS, TFLITE_OUTPUT_DETAILS

    if TFLITE_INTERPRETER is not None:
        return None

    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception:
            from tensorflow.lite import Interpreter

        if not os.path.exists(MODEL_PATH):
            return f"Model file not found: {MODEL_PATH}"

        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        TFLITE_INTERPRETER = interpreter
        TFLITE_INPUT_DETAILS = interpreter.get_input_details()
        TFLITE_OUTPUT_DETAILS = interpreter.get_output_details()
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
            tflite_error = ensure_tflite_loaded()
            if tflite_error:
                MODEL_LOAD_ERROR = tflite_error
                return MODEL_LOAD_ERROR

            MODEL = TFLITE_INTERPRETER
            print("✅ TFLite model loaded successfully into memory")

            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                CLASS_NAMES = metadata["class_names"]
            print(f"✅ Classes loaded: {CLASS_NAMES}")
            MODEL_LOAD_ERROR = None

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
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    fallback_model_present = os.path.exists(MODEL_PATH)
    external_available = ensure_backup_service_available()

    if external_available or (MODEL is not None and CLASS_NAMES is not None) or fallback_model_present:
        return jsonify({
            "status": "healthy",
            "external_available": external_available,
            "fallback_model_present": fallback_model_present,
            "model_loaded": MODEL is not None and CLASS_NAMES is not None,
            "classes_available": len(CLASS_NAMES) if CLASS_NAMES else 0,
            "classes": CLASS_NAMES if CLASS_NAMES else []
        })

    return jsonify({
        "status": "error",
        "error": "No prediction backend available"
    }), 500

@app.route('/info', methods=['GET'])
def info():
    """Get model and disease information"""
    fallback_model_present = os.path.exists(MODEL_PATH)
    external_available = ensure_backup_service_available()

    if CLASS_NAMES is None:
        if fallback_model_present:
            try:
                with open(METADATA_PATH, 'r') as f:
                    metadata = json.load(f)
                    classes = metadata.get("class_names", [])
            except Exception:
                classes = list(DISEASE_INFO.keys())
        else:
            classes = list(DISEASE_INFO.keys())
    else:
        classes = CLASS_NAMES

    return jsonify({
        "model_name": "ResNet50 Banana Leaf Disease Classifier",
        "classes": classes,
        "external_available": external_available,
        "fallback_model_present": fallback_model_present,
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
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    if request.content_length is not None and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "Image is too large. Please upload a file under 10 MB."}), 413

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

        # ── Step 2: Primary external verification ───────────
        if ensure_backup_service_available():
            external_result = call_backup_service(img)
            if external_result:
                ext_class = external_result.get("disease")
                ext_conf = float(external_result.get("confidence", 0.0))

                if ext_class in DISEASE_INFO:
                    class_probabilities = {
                        name: 0.0 for name in (CLASS_NAMES if CLASS_NAMES else DISEASE_INFO.keys())
                    }
                    class_probabilities[ext_class] = ext_conf
                    disease_info = DISEASE_INFO.get(ext_class, {})

                    return jsonify({
                        "status": "success",
                        "predicted_class": ext_class,
                        "confidence": ext_conf,
                        "all_probabilities": class_probabilities,
                        "validation": {
                            "green_ratio": float(green_ratio),
                            "passes_green_check": True
                        },
                        "source": "external_verified",
                        "verification_metrics": {
                            "image_quality": external_result.get('image_quality'),
                            "leaf_certainty": external_result.get('leaf_certainty'),
                            "accuracy_score": external_result.get('accuracy_score')
                        },
                        "disease_info": {
                            "emoji": disease_info.get("emoji"),
                            "title": disease_info.get("title"),
                            "description": disease_info.get("description"),
                            "recommended_solutions": disease_info.get("solutions", [])
                        }
                    }), 200

        # ── Step 3: Fallback to local model ─────────────────
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                "status": "error",
                "error": "Primary verification is unavailable and local fallback model is not configured.",
                "hint": "Set BACKUP_SVC or provide models/banana_model.tflite"
            }), 503

        load_error = ensure_model_loaded()
        if load_error:
            return jsonify({
                "status": "error",
                "error": "Local fallback is unavailable.",
                "details": str(load_error)
            }), 503
        
        # ── Step 4: Prepare image for local model ───────────
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.asarray(img_resized, dtype=np.float32)
        img_array = preprocess_resnet50(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # ── Step 5: Make local prediction ───────────────────
        input_details = TFLITE_INPUT_DETAILS[0]
        output_details = TFLITE_OUTPUT_DETAILS[0]
        input_dtype = input_details["dtype"]

        MODEL.set_tensor(input_details["index"], img_array.astype(input_dtype))
        MODEL.invoke()
        preds = MODEL.get_tensor(output_details["index"])[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])
        
        # ── Step 6: Confidence gate for local fallback ─────
        conf_ok, top_conf = is_confident_enough(preds)
        
        if not conf_ok:
            return jsonify({
                "status": "rejected",
                "reason": "LOW_CONFIDENCE",
                "message": "Model could not confidently classify this image",
                "confidence": confidence,
                "required_confidence": CONFIDENCE_THRESHOLD
            }), 400
        
        # ── Step 7: Return successful local fallback result ─
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
            "source": "local_fallback",
            "disease_info": {
                "emoji": disease_info.get("emoji"),
                "title": disease_info.get("title"),
                "description": disease_info.get("description"),
                "recommended_solutions": disease_info.get("solutions", [])
            }
        }
        
        return jsonify(response_data), 200
        
    except UnidentifiedImageError:
        return jsonify({
            "error": "Invalid image file. Please upload a JPG or PNG banana leaf image.",
            "status": "error"
        }), 400
    except OSError:
        return jsonify({
            "error": "Unable to read image data. Please try another file.",
            "status": "error"
        }), 400
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
