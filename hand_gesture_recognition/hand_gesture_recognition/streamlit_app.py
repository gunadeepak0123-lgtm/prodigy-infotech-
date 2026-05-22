import streamlit as st
import pickle
import numpy as np
import json
import os
from PIL import Image
from skimage.feature import hog

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
IMAGE_SIZE = (64, 64)

# ── Load model ─────────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'model.pkl'),   'rb') as f: model   = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'),  'rb') as f: scaler  = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'rb') as f: encoder = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'metrics.json'))      as f: metrics = json.load(f)

GESTURE_EMOJIS = {
    'Palm': '🖐️', 'Fist': '✊', 'Thumb Up': '👍', 'Index': '☝️',
    'OK': '👌', 'Peace': '✌️', 'C': '🤏', 'L': '🤙',
    'Down': '👇', 'Palm Moved': '🖐️', 'Fist Moved': '✊',
}

# ── Feature extraction — grayscale HOG, always 1764 features ──────────────────
def extract_features(img: Image.Image):
    img_gray  = img.convert('L').resize(IMAGE_SIZE)   # grayscale 64x64
    img_array = np.array(img_gray)                     # shape: (64, 64)
    features  = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
        # no channel_axis — grayscale image
    )
    return features   # always 1764 features

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤚")
st.title("🤚 Hand Gesture Recognition")
st.caption("Prodigy Infotech — Task 04 · Random Forest + HOG Features")
st.divider()

# ── Stats ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model",     "Random Forest")
col2.metric("Accuracy",  f"{metrics['accuracy']}%")
col3.metric("Gestures",  metrics['n_classes'])
col4.metric("Data",      "Real" if metrics['using_real'] else "Demo")

if not metrics['using_real']:
    st.info("ℹ️ Running in Demo Mode. Predictions may not be accurate on real images.")

st.divider()

# ── Supported gestures ─────────────────────────────────────────────────────────
st.subheader("🖐️ Supported Gestures")
gestures = metrics['gestures']
cols = st.columns(5)
for i, gesture in enumerate(gestures):
    emoji = GESTURE_EMOJIS.get(gesture, '🤚')
    cols[i % 5].markdown(f"**{emoji} {gesture}**")

st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
st.subheader("📤 Upload a Hand Gesture Image")

uploaded_file = st.file_uploader(
    "Choose an image of a hand gesture",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.write("**Image Info:**")
        st.write(f"- Original size: {img.size[0]} × {img.size[1]} px")
        st.write(f"- Converted to: Grayscale")
        st.write(f"- Resized to: 64 × 64 px")
        st.write(f"- HOG features: 1764")

    st.divider()

    if st.button("🔍 Recognize Gesture", type="primary", use_container_width=True):
        with st.spinner("Analyzing gesture..."):
            features        = extract_features(img)
            features_scaled = scaler.transform([features])
            prediction      = model.predict(features_scaled)[0]
            proba           = model.predict_proba(features_scaled)[0]
            label           = encoder.inverse_transform([prediction])[0]
            confidence      = proba[prediction] * 100
            emoji           = GESTURE_EMOJIS.get(label, '🤚')

            st.subheader("Result")
            st.success(f"### {emoji} {label}")
            st.metric("Confidence", f"{confidence:.1f}%")

            st.divider()
            st.write("**Top 3 Predictions:**")
            top3 = np.argsort(proba)[::-1][:3]
            for rank, idx in enumerate(top3):
                g_label = encoder.inverse_transform([idx])[0]
                g_emoji = GESTURE_EMOJIS.get(g_label, '🤚')
                g_conf  = proba[idx] * 100
                st.write(f"{rank+1}. {g_emoji} **{g_label}** — `{g_conf:.1f}%`")
                st.progress(float(proba[idx]))

st.divider()
st.subheader("📦 Kaggle Dataset")
st.markdown("[👉 Download LeapGestRecog Dataset](https://www.kaggle.com/gti-upm/leapgestrecog)")
st.write("""
**To use real data:**
1. Download and unzip the dataset
2. Place gesture folders inside `data/train/`
3. Run `python model/train_model.py`
4. Restart the app
""")
