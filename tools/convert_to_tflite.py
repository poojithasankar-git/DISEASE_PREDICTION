import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(BASE_DIR, "models", "banana_model.h5")
TFLITE_PATH = os.path.join(BASE_DIR, "models", "banana_model.tflite")

if not os.path.exists(H5_PATH):
    raise FileNotFoundError(f"Missing model: {H5_PATH}")

model = tf.keras.models.load_model(H5_PATH, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable default optimizations to reduce size.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save.
tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved: {TFLITE_PATH}")
