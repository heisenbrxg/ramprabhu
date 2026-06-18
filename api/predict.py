import io
import os
import urllib.request

import numpy as np
from flask import Flask, jsonify, make_response, request
from PIL import Image

app = Flask(__name__)

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_COLORS = {
    "NonDemented": "#2ecc71",
    "VeryMildDemented": "#f39c12",
    "MildDemented": "#e67e22",
    "ModerateDemented": "#e74c3c",
}
CLASS_DESCRIPTIONS = {
    "NonDemented": "No signs of dementia detected.",
    "VeryMildDemented": "Very mild cognitive decline detected.",
    "MildDemented": "Mild stage of Alzheimer's detected.",
    "ModerateDemented": "Moderate stage of Alzheimer's detected.",
}
IMG_SIZE = 224
_session = None
DEMO_MODE = not os.environ.get("MODEL_URL", "")


def get_session():
    global _session
    if _session is not None:
        return _session

    import onnxruntime as ort

    cached = "/tmp/model.onnx"
    model_url = os.environ.get("MODEL_URL", "")

    if not os.path.exists(cached):
        if model_url:
            urllib.request.urlretrieve(model_url, cached)
        else:
            raise FileNotFoundError("MODEL_URL not set")

    _session = ort.InferenceSession(cached)
    return _session


def demo_predict():
    """Return plausible-looking fake probabilities for demo/screenshot use."""
    raw = np.random.dirichlet([1, 0.3, 4, 2])  # skewed toward NonDemented
    return raw.tolist()


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/api/predict", methods=["OPTIONS"])
def predict_preflight():
    return make_response("", 204)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        if DEMO_MODE:
            probs = demo_predict()
        else:
            img = Image.open(io.BytesIO(f.read())).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
            session = get_session()
            input_name = session.get_inputs()[0].name
            probs = session.run(None, {input_name: arr})[0][0].tolist()

        predicted_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[predicted_idx]

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(probs[predicted_idx] * 100, 2),
            "color": CLASS_COLORS[predicted_class],
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "probabilities": {
                cls: round(p * 100, 2) for cls, p in zip(CLASS_NAMES, probs)
            },
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    if DEMO_MODE:
        return jsonify({"status": "ok"})
    try:
        get_session()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503
