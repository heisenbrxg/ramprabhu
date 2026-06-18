import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="centered",
)

MODEL_PATH = "alzheimer_model.h5"
IMG_SIZE = 224
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_DESCRIPTIONS = {
    "NonDemented": "No signs of dementia detected.",
    "VeryMildDemented": "Very mild cognitive decline detected.",
    "MildDemented": "Mild stage of Alzheimer's detected.",
    "ModerateDemented": "Moderate stage of Alzheimer's detected.",
}
CLASS_COLORS = {
    "NonDemented": "#2ecc71",
    "VeryMildDemented": "#f39c12",
    "MildDemented": "#e67e22",
    "ModerateDemented": "#e74c3c",
}

# ── Model loader (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Alzheimer's MRI Image Classifier")
st.markdown(
    "Upload a brain MRI scan to classify the stage of Alzheimer's disease "
    "using a deep learning model trained on the Kaggle Alzheimer MRI dataset."
)

st.warning(
    "⚠️ **Medical Disclaimer:** This tool is for **educational and research purposes only**. "
    "It is **not a substitute** for professional medical diagnosis, advice, or treatment. "
    "Always consult a qualified healthcare provider for any medical concerns."
)

st.divider()

# ── Model status ──────────────────────────────────────────────────────────────
model = load_model()
if model is None:
    st.error(
        f"Model file `{MODEL_PATH}` not found. "
        "Please run `python train.py` first to train and save the model."
    )
    st.stop()

st.success("Model loaded successfully.")

# ── File uploader ─────────────────────────────────────────────────────────────
st.subheader("Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose a brain MRI image (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with col2:
        st.markdown("**Image details**")
        st.write(f"- Format: `{image.format or 'N/A'}`")
        st.write(f"- Size: `{image.width} × {image.height} px`")
        st.write(f"- Mode: `{image.mode}`")

    # ── Prediction ────────────────────────────────────────────────────────────
    with st.spinner("Analysing image..."):
        input_arr = preprocess_image(image)
        probs = model.predict(input_arr, verbose=0)[0]

    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx]) * 100

    st.divider()
    st.subheader("Prediction Result")

    color = CLASS_COLORS[predicted_class]
    st.markdown(
        f"<div style='background:{color}22; border-left:5px solid {color}; "
        f"padding:16px; border-radius:6px;'>"
        f"<h3 style='color:{color}; margin:0;'>{predicted_class}</h3>"
        f"<p style='margin:4px 0 0 0;'>{CLASS_DESCRIPTIONS[predicted_class]}</p>"
        f"<p style='margin:4px 0 0 0;'><strong>Confidence: {confidence:.1f}%</strong></p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Confidence bar chart ──────────────────────────────────────────────────
    st.subheader("Confidence Scores — All Classes")

    bar_colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    fig = go.Figure(
        go.Bar(
            x=CLASS_NAMES,
            y=[float(p) * 100 for p in probs],
            marker_color=bar_colors,
            text=[f"{float(p)*100:.1f}%" for p in probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Confidence (%)", range=[0, 110]),
        xaxis=dict(title="Class"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw scores table
    with st.expander("Raw probability scores"):
        for cls, prob in zip(CLASS_NAMES, probs):
            st.write(f"**{cls}**: {float(prob):.4f}")

st.divider()
st.caption(
    "Model: EfficientNetB0 fine-tuned on the Alzheimer MRI Preprocessed Dataset (Kaggle). "
    "This application is built with Streamlit and TensorFlow."
)
