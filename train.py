# ============================================================
#   BANANA LEAF DISEASE CLASSIFICATION — TRAINING SCRIPT
#   ResNet50 | Download → Train → Evaluate → Save Model
#   Optimized for Render (Run once, then serve)
# ============================================================

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import kagglehub

# ── Configuration ────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 10
MODEL_PATH = "models/banana_disease_model.h5"
METADATA_PATH = "models/class_metadata.json"

# Create models directory
os.makedirs("models", exist_ok=True)

# Create log file
log_file = open("training_log.txt", 'w')

def log_print(msg):
    """Print to both console and log file"""
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log_print("=" * 70)
log_print("🚀 BANANA LEAF DISEASE CLASSIFICATION - TRAINING START")
log_print("=" * 70)

# ── 1. Download Dataset from Kaggle ──────────────────────────
log_print("\n📥 Downloading dataset from Kaggle...")
try:
    dataset_root = kagglehub.dataset_download("shifatearman/bananalsd")
    dataset_path = os.path.join(dataset_root, "BananaLSD", "OriginalSet")
    log_print(f"✅ Dataset downloaded: {dataset_path}")
except Exception as e:
    log_print(f"❌ Error downloading dataset: {e}")
    log_print("Make sure KAGGLE_API_TOKEN is set (or KAGGLE_USERNAME and KAGGLE_KEY)")
    sys.exit(1)

# ── 2. Verify dataset exists ─────────────────────────────────
if not os.path.exists(dataset_path):
    log_print(f"❌ Dataset path not found: {dataset_path}")
    sys.exit(1)

# ── 3. Set up Data Generators ────────────────────────────────
log_print("\n🔄 Setting up data generators...")

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())

log_print(f"✅ Data generators ready")
log_print(f"   Classes: {class_names}")
log_print(f"   Training samples: {train_gen.samples}")
log_print(f"   Validation samples: {val_gen.samples}")

# ── 4. Build Model (ResNet50 + Custom Head) ──────────────────
log_print("\n🏗️  Building ResNet50 model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = Dense(256, activation="relu")(x)
x      = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

log_print("✅ Model compiled successfully")

# ── 5. Train ─────────────────────────────────────────────────
log_print(f"\n🚀 Starting training ({EPOCHS} epochs)...")
log_print("   This may take 30-60 minutes on Render free tier...")

try:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )
    log_print("\n✅ Training complete!")
except Exception as e:
    log_print(f"\n❌ Training error: {e}")
    sys.exit(1)

# ── 6. Evaluate on Validation Set ────────────────────────────
log_print("\n📈 Evaluating model...")

y_true       = val_gen.classes
y_pred_probs = model.predict(val_gen, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

log_print("\n" + "=" * 70)
log_print("📊 EVALUATION RESULTS")
log_print("=" * 70)
log_print(f"  Accuracy  : {accuracy:.4f}")
log_print(f"  Precision : {precision:.4f}")
log_print(f"  Recall    : {recall:.4f}")
log_print(f"  F1 Score  : {f1:.4f}")
log_print("=" * 70)

log_print("\n📋 Classification Report:\n")
report = classification_report(y_true, y_pred, target_names=class_names)
log_print(report)

# ── 7. Save Model ────────────────────────────────────────────
log_print(f"\n💾 Saving model to {MODEL_PATH}...")
try:
    model.save(MODEL_PATH)
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    log_print(f"✅ Model saved successfully ({file_size:.1f} MB)")
except Exception as e:
    log_print(f"❌ Error saving model: {e}")
    sys.exit(1)

# ── 8. Save Metadata (class names) ───────────────────────────
log_print(f"\n📝 Saving metadata to {METADATA_PATH}...")
metadata = {
    "class_names": class_names,
    "num_classes": num_classes,
    "img_size": IMG_SIZE,
    "accuracy": float(accuracy),
    "f1_score": float(f1),
    "training_epochs": EPOCHS
}

try:
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)
    log_print(f"✅ Metadata saved successfully")
except Exception as e:
    log_print(f"❌ Error saving metadata: {e}")

# ── 9. Summary ───────────────────────────────────────────────
log_print("\n" + "=" * 70)
log_print("✅ TRAINING PIPELINE COMPLETE")
log_print("=" * 70)
log_print(f"\n📂 Output files:")
log_print(f"   - models/{MODEL_PATH} (trained model)")
log_print(f"   - models/{METADATA_PATH} (class metadata)")
log_print(f"   - training_log.txt (this log)")
log_print(f"\n🎯 Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
log_print(f"\n📝 Model is ready to be loaded by the Flask API server")
log_print(f"   Server will load model into memory and keep it there until restart")
log_print("=" * 70)

log_file.close()
print(f"\n✅ Check 'training_log.txt' for complete training details")
