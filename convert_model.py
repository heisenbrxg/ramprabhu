# Run after training: python convert_model.py
# Then upload alzheimer_model.onnx and set MODEL_URL in Vercel dashboard.

import os
import sys

MODEL_H5   = "alzheimer_model.h5"
MODEL_ONNX = "alzheimer_model.onnx"
IMG_SIZE   = 224

if not os.path.exists(MODEL_H5):
    print(f"ERROR: {MODEL_H5} not found. Run train.py first.")
    sys.exit(1)

try:
    import tensorflow as tf
    import tf2onnx
except ImportError:
    print("Install: pip install tf2onnx")
    sys.exit(1)

print("Loading model…")
model = tf.keras.models.load_model(MODEL_H5)

print("Converting to ONNX…")
sig = [tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="input")]
proto, _ = tf2onnx.convert.from_keras(model, input_signature=sig, opset=13)

with open(MODEL_ONNX, "wb") as f:
    f.write(proto.SerializeToString())

size_mb = os.path.getsize(MODEL_ONNX) / 1024 / 1024
print(f"Saved {MODEL_ONNX}  ({size_mb:.1f} MB)")
print()
print("Next steps:")
print("  1. Upload alzheimer_model.onnx to a public URL, e.g.:")
print("       Hugging Face Hub  → https://huggingface.co/new")
print("       GitHub Releases   → works for files up to 2 GB")
print("  2. Vercel dashboard → Project → Settings → Environment Variables:")
print("       MODEL_URL = <your direct download URL>")
print("  3. vercel --prod")
