import streamlit as st
import pickle
import numpy as np
import json
import os

# ── Load model ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'model')

with open(os.path.join(MODEL_DIR, 'model.pkl'),   'rb') as f: model  = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'),  'rb') as f: scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'metrics.json'))      as f: metrics = json.load(f)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.caption("Prodigy Infotech — Task 01 · Linear Regression Model")

st.divider()

# ── Model stats ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("R² Accuracy",       f"{metrics['r2']*100:.1f}%")
col2.metric("Avg. Error (RMSE)", f"${metrics['rmse']:,.0f}")
col3.metric("Training Samples",  f"{metrics['n_samples']:,}")

st.divider()

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.subheader("Enter Property Details")

sqft      = st.number_input("Square Footage (sq ft)", min_value=100, max_value=15000, value=1500, step=50)
bedrooms  = st.slider("Number of Bedrooms",  min_value=1, max_value=10, value=3)
bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=8,  value=2)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Price", use_container_width=True, type="primary"):
    X = np.array([[sqft, bedrooms, bathrooms]])
    X_scaled = scaler.transform(X)
    price = model.predict(X_scaled)[0]
    price = max(price, 10000)

    low  = price * 0.85
    high = price * 1.15

    st.divider()
    st.subheader("Estimated Price")
    st.metric("Predicted Sale Price", f"${price:,.0f}")
    st.write(f"**Estimated Range:** ${low:,.0f} — ${high:,.0f}")

    st.divider()
    st.write("**Your Input Summary:**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Square Footage", f"{sqft:,} sq ft")
    c2.metric("Bedrooms",  bedrooms)
    c3.metric("Bathrooms", bathrooms)
