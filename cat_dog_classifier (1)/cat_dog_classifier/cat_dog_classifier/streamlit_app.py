import streamlit as st
import pickle
import numpy as np
import json
import os
from PIL import Image
from skimage.feature import hog

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
IMAGE_SIZE  = (64, 64)

# ── Load model ─────────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'rb') as f: model  = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'),    'rb') as f: scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'metrics.json'))        as f: metrics = json.load(f)

# ── Feature extraction — always HOG, always 1568 features ─────────────────────
def extract_features(img: Image.Image):
    img_rgb    = img.convert('RGB').resize(IMAGE_SIZE)  # resize to 64x64
    img_array  = np.array(img_rgb)                       # shape: (64, 64, 3)
    features   = hog(
        img_array,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1                                  # RGB image
    )
    return features   # always 1568 features

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
st.title("🐾 Cat vs Dog Classifier")
st.caption("Prodigy Infotech — Task 03 · SVM + HOG Features")
st.divider()

# ── Stats ──────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Model",     "SVM (RBF Kernel)")
col2.metric("Accuracy",  f"{metrics['accuracy']}%")
col3.metric("Data Mode", "Real Images" if metrics['using_real'] else "Demo (Synthetic)")

if not metrics['using_real']:
    st.info("""
    ℹ️ Running in Demo Mode (synthetic training data).
    To use real images, place cat images in `data/train/cats/` and dog images in `data/train/dogs/`,
    then run `python model/train_model.py`.
    """)

st.divider()

# ── How it works ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ How does this work?"):
    st.write("""
    1. Your image is resized to **64 × 64 pixels**
    2. **HOG features** are extracted (1568 values) — captures edges and texture
    3. Features are **standardised** with StandardScaler
    4. **SVM with RBF kernel** predicts Cat or Dog
    5. **Probability score** shows model confidence
    """)

st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
st.subheader("📤 Upload a Cat or Dog Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Your Image", use_container_width=True)

    with col2:
        st.write("**Image Info:**")
        st.write(f"- Original size: {img.size[0]} × {img.size[1]} px")
        st.write(f"- Will be resized to: 64 × 64 px")
        st.write(f"- HOG features extracted: 1568")

    st.divider()

    if st.button("🔍 Classify Image", type="primary", use_container_width=True):
        with st.spinner("Classifying..."):
            features        = extract_features(img)           # 1568 HOG features
            features_scaled = scaler.transform([features])    # scale
            prediction      = model.predict(features_scaled)[0]
            proba           = model.predict_proba(features_scaled)[0]

            label      = "Dog 🐶" if prediction == 1 else "Cat 🐱"
            confidence = proba[prediction] * 100

            st.subheader("Result")
            if prediction == 1:
                st.success(f"### {label}")
            else:
                st.info(f"### {label}")

            st.metric("Confidence", f"{confidence:.1f}%")

            st.write("**Probability Breakdown:**")
            st.write(f"🐱 Cat: `{proba[0]*100:.1f}%`")
            st.progress(float(proba[0]))
            st.write(f"🐶 Dog: `{proba[1]*100:.1f}%`")
            st.progress(float(proba[1]))

st.divider()
st.subheader("📦 Kaggle Dataset")
st.markdown("[👉 Download Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)")
st.write("""
**To use real data:**
1. Download and unzip the dataset
2. Place cat images → `data/train/cats/`
3. Place dog images → `data/train/dogs/`
4. Run `python model/train_model.py`
5. Restart the app
""")
