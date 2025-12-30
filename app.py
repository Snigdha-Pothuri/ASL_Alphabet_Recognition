import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image
import gzip
import shutil
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="ASL Alphabet Recognition",
    page_icon="🤟",
    layout="wide"
)

# -------------------------------
# Custom CSS (Cream + Purple Theme)
# -------------------------------
st.markdown("""
<style>
/* App Background */
.stApp {
    background-color: #FFF8F0;
    font-family: 'Segoe UI', sans-serif;
}

.hero-title {
    text-align: center;
    font-size: 60px;
    font-weight: 900;
    background: linear-gradient(135deg, #861657 0%, #AA4465 50%, #C15B8E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 12px rgba(134, 22, 87, 0.3);
}

.hero-tagline {
    text-align: center;
    font-size: 24px;
    color: #861657;
    font-weight: 600;
    margin-bottom: 50px;
    letter-spacing: 1px;
}

.letter-circle {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: linear-gradient(135deg, #861657 0%, #AA4465 100%);
    color: white;
    font-size: 110px;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 40px auto;
    box-shadow: 0 25px 50px rgba(134, 22, 87, 0.4);
}

.confidence-badge {
    position: absolute;
    bottom: 15px;
    right: 15px;
    background: rgba(255,255,255,0.95);
    color: #4A1F3E;
    padding: 8px 14px;
    border-radius: 25px;
    font-size: 18px;
    font-weight: 800;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

.label {
    color: #5A3B5C;
    font-weight: 600;
    margin: 8px 0 4px 0;
    font-size: 20px;
}

div[data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] [data-testid="StyledFileDropzoneFileName"] {
    color: #2D1B3A !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

label, .stFileUploader label {
    color: #861657 !important;
    font-weight: 600;
}

div[data-testid="stAlert"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Custom function for purple progress bars
# -------------------------------
def purple_progress_bar(label, value):
    st.markdown(f"""
    <div style="margin:6px 0;">
        <div style="font-weight:600; color:#5A3B5C;">{label}: {value:.1f}%</div>
        <div style="background:#F5E8F0; border-radius:4px; height:8px;">
            <div style="width:{value}%; background:#861657; height:100%; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Title & Tagline
# -------------------------------
st.markdown("""
<div class='hero-title'>
    <span style='-webkit-text-fill-color: initial; color: initial;'>🖐️</span> ASL Alphabet Recognition
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='hero-tagline'>Identifies hand gestures and shows the top 3 predicted letters with confidence scores</div>", unsafe_allow_html=True)

# -------------------------------
# Load Model from gzipped file
# -------------------------------
@st.cache_resource
def load_asl_model():
    compressed_file = "asl_mobilenet_model.h5.gz"
    temp_file = "asl_temp_model.h5"

    with gzip.open(compressed_file, 'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    model = load_model(temp_file)
    os.remove(temp_file)
    return model

model = load_asl_model()
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload ASL Hand Gesture Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Only render UI AFTER upload
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1], gap="large")

    # Left: Image
    with col1:
        st.image(image, caption="Uploaded Image", width=500)

    # Right: Predictions
    with col2:
        IMG_SIZE = 128
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array, verbose=0)[0]
        top3_idx = preds.argsort()[-3:][::-1]
        top3 = [(class_labels[i], preds[i] * 100) for i in top3_idx]

        # Huge Letter Circle with Confidence
        st.markdown(
            f"""
            <div class='letter-circle' style='position: relative;'>
    {top3[0][0]}
    <div class='confidence-badge'>{top3[0][1]:.1f}%</div>
</div>
            """,
            unsafe_allow_html=True
        )

        # Top 3 Predictions
        st.markdown("<h3 style='color:#861657; text-align:center;'>Top 3 Matches</h3>", unsafe_allow_html=True)
        for i, (letter, conf) in enumerate(top3, 1):
            purple_progress_bar(f"{i}. {letter}", conf)
