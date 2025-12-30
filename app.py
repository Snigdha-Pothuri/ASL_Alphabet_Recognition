import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image

# -------------------------------
# Page config
st.set_page_config(page_title="ASL Alphabet Recognition", page_icon="🤟", layout="wide")

# -------------------------------
# Load model from root
MODEL_PATH = os.path.join(os.path.dirname(__file__), "asl_mobilenet_model.h5")

@st.cache_resource
def load_asl_model():
    return load_model(MODEL_PATH)

model = load_asl_model()
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -------------------------------
# Custom CSS
st.markdown("""
<style>
.stApp {background-color: #FFF8F0; font-family: 'Segoe UI', sans-serif;}
.hero-title {text-align:center; font-size:60px; font-weight:900; 
background:linear-gradient(135deg,#861657 0%,#AA4465 50%,#C15B8E 100%); 
-webkit-background-clip:text; -webkit-text-fill-color:transparent; 
text-shadow:0 4px 12px rgba(134,22,87,0.3);}
.hero-tagline {text-align:center; font-size:24px; color:#861657; font-weight:600; margin-bottom:50px;}
.letter-circle {width:220px; height:220px; border-radius:50%; background:linear-gradient(135deg,#861657 0%,#AA4465 100%); color:white; font-size:110px; font-weight:bold; display:flex; align-items:center; justify-content:center; margin:40px auto; box-shadow:0 25px 50px rgba(134,22,87,0.4);}
.confidence-badge {position:absolute; bottom:15px; right:15px; background:rgba(255,255,255,0.95); color:#4A1F3E; padding:8px 14px; border-radius:25px; font-size:18px; font-weight:800;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Purple progress bar
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
# Title & tagline
st.markdown("<div class='hero-title'>🖐️ ASL Alphabet Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-tagline'>Identifies hand gestures and shows the top 3 predictions</div>", unsafe_allow_html=True)

# -------------------------------
# Upload image
uploaded_file = st.file_uploader("📤 Upload ASL Hand Gesture Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width=450)
    with col2:
        IMG_SIZE = 128
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = preprocess_input(img_array)
        preds = model.predict(img_array, verbose=0)[0]
        top3_idx = preds.argsort()[-3:][::-1]
        top3 = [(class_labels[i], preds[i]*100) for i in top3_idx]
        # Letter circle
        st.markdown(f"""
            <div class='letter-circle' style='position:relative;'>
                {top3[0][0]}
                <div class='confidence-badge'>{top3[0][1]:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
        # Top 3 progress
        st.markdown("<h3 style='color:#861657; text-align:center;'>Top 3 Matches</h3>", unsafe_allow_html=True)
        for i, (letter, conf) in enumerate(top3,1):
            purple_progress_bar(f"{i}. {letter}", conf)
