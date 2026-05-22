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

# ── Full calorie database ──────────────────────────────────────────────────────
CALORIE_DB = {
    'apple_pie':           {'calories': 237, 'protein': 2,  'carbs': 34, 'fat': 11, 'serving': '1 slice (125g)'},
    'baby_back_ribs':      {'calories': 297, 'protein': 26, 'carbs': 0,  'fat': 21, 'serving': '100g'},
    'baklava':             {'calories': 428, 'protein': 6,  'carbs': 52, 'fat': 23, 'serving': '1 piece (85g)'},
    'beef_carpaccio':      {'calories': 150, 'protein': 20, 'carbs': 1,  'fat': 8,  'serving': '100g'},
    'bibimbap':            {'calories': 490, 'protein': 26, 'carbs': 69, 'fat': 12, 'serving': '1 bowl'},
    'bread_pudding':       {'calories': 153, 'protein': 5,  'carbs': 25, 'fat': 4,  'serving': '100g'},
    'breakfast_burrito':   {'calories': 305, 'protein': 14, 'carbs': 30, 'fat': 14, 'serving': '1 burrito'},
    'bruschetta':          {'calories': 194, 'protein': 6,  'carbs': 28, 'fat': 7,  'serving': '100g'},
    'caesar_salad':        {'calories': 158, 'protein': 4,  'carbs': 8,  'fat': 13, 'serving': '1 bowl'},
    'cannoli':             {'calories': 225, 'protein': 5,  'carbs': 26, 'fat': 12, 'serving': '1 piece'},
    'caprese_salad':       {'calories': 166, 'protein': 10, 'carbs': 4,  'fat': 12, 'serving': '100g'},
    'carrot_cake':         {'calories': 415, 'protein': 4,  'carbs': 55, 'fat': 21, 'serving': '1 slice'},
    'cheesecake':          {'calories': 321, 'protein': 5,  'carbs': 26, 'fat': 23, 'serving': '1 slice'},
    'chicken_curry':       {'calories': 150, 'protein': 12, 'carbs': 7,  'fat': 8,  'serving': '100g'},
    'chicken_wings':       {'calories': 290, 'protein': 27, 'carbs': 0,  'fat': 19, 'serving': '100g'},
    'chocolate_cake':      {'calories': 371, 'protein': 5,  'carbs': 51, 'fat': 17, 'serving': '1 slice'},
    'club_sandwich':       {'calories': 590, 'protein': 35, 'carbs': 40, 'fat': 29, 'serving': '1 sandwich'},
    'donuts':              {'calories': 452, 'protein': 5,  'carbs': 51, 'fat': 25, 'serving': '1 donut'},
    'dumplings':           {'calories': 336, 'protein': 14, 'carbs': 33, 'fat': 17, 'serving': '6 pieces'},
    'edamame':             {'calories': 122, 'protein': 11, 'carbs': 10, 'fat': 5,  'serving': '100g'},
    'eggs_benedict':       {'calories': 590, 'protein': 24, 'carbs': 30, 'fat': 39, 'serving': '1 serving'},
    'falafel':             {'calories': 333, 'protein': 13, 'carbs': 32, 'fat': 18, 'serving': '100g'},
    'french_fries':        {'calories': 312, 'protein': 4,  'carbs': 41, 'fat': 15, 'serving': '100g'},
    'french_toast':        {'calories': 311, 'protein': 9,  'carbs': 36, 'fat': 14, 'serving': '2 slices'},
    'fried_rice':          {'calories': 163, 'protein': 4,  'carbs': 29, 'fat': 3,  'serving': '100g'},
    'frozen_yogurt':       {'calories': 159, 'protein': 4,  'carbs': 34, 'fat': 2,  'serving': '100g'},
    'greek_salad':         {'calories': 211, 'protein': 6,  'carbs': 8,  'fat': 17, 'serving': '1 bowl'},
    'hamburger':           {'calories': 540, 'protein': 34, 'carbs': 40, 'fat': 26, 'serving': '1 burger'},
    'hot_dog':             {'calories': 290, 'protein': 11, 'carbs': 24, 'fat': 17, 'serving': '1 hot dog'},
    'hummus':              {'calories': 177, 'protein': 8,  'carbs': 20, 'fat': 9,  'serving': '100g'},
    'ice_cream':           {'calories': 207, 'protein': 4,  'carbs': 24, 'fat': 11, 'serving': '1 scoop'},
    'lasagna':             {'calories': 135, 'protein': 8,  'carbs': 13, 'fat': 5,  'serving': '100g'},
    'macaroni_and_cheese': {'calories': 164, 'protein': 6,  'carbs': 22, 'fat': 5,  'serving': '100g'},
    'nachos':              {'calories': 346, 'protein': 9,  'carbs': 36, 'fat': 19, 'serving': '100g'},
    'omelette':            {'calories': 154, 'protein': 11, 'carbs': 1,  'fat': 12, 'serving': '1 omelette'},
    'onion_rings':         {'calories': 411, 'protein': 6,  'carbs': 43, 'fat': 24, 'serving': '100g'},
    'pad_thai':            {'calories': 225, 'protein': 12, 'carbs': 32, 'fat': 6,  'serving': '100g'},
    'pancakes':            {'calories': 227, 'protein': 6,  'carbs': 30, 'fat': 9,  'serving': '2 pancakes'},
    'pho':                 {'calories': 215, 'protein': 15, 'carbs': 28, 'fat': 5,  'serving': '1 bowl'},
    'pizza':               {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10, 'serving': '1 slice'},
    'ramen':               {'calories': 436, 'protein': 21, 'carbs': 57, 'fat': 14, 'serving': '1 bowl'},
    'samosa':              {'calories': 262, 'protein': 5,  'carbs': 28, 'fat': 15, 'serving': '2 pieces'},
    'sashimi':             {'calories': 127, 'protein': 20, 'carbs': 0,  'fat': 5,  'serving': '100g'},
    'spaghetti_bolognese': {'calories': 300, 'protein': 15, 'carbs': 38, 'fat': 9,  'serving': '1 bowl'},
    'spaghetti_carbonara': {'calories': 370, 'protein': 17, 'carbs': 40, 'fat': 16, 'serving': '1 bowl'},
    'spring_rolls':        {'calories': 154, 'protein': 4,  'carbs': 18, 'fat': 7,  'serving': '2 rolls'},
    'steak':               {'calories': 271, 'protein': 26, 'carbs': 0,  'fat': 18, 'serving': '100g'},
    'sushi':               {'calories': 143, 'protein': 6,  'carbs': 24, 'fat': 2,  'serving': '6 pieces'},
    'tacos':               {'calories': 226, 'protein': 12, 'carbs': 21, 'fat': 10, 'serving': '2 tacos'},
    'tiramisu':            {'calories': 240, 'protein': 5,  'carbs': 28, 'fat': 12, 'serving': '1 slice'},
    'waffles':             {'calories': 291, 'protein': 8,  'carbs': 37, 'fat': 13, 'serving': '1 waffle'},
}

def get_calorie_info(food_key):
    return CALORIE_DB.get(food_key, {'calories': 200, 'protein': 8, 'carbs': 25, 'fat': 8, 'serving': '100g'})

def get_calorie_color(calories):
    if calories < 200:   return "🟢 Low"
    elif calories < 400: return "🟡 Medium"
    else:                return "🔴 High"

# ── Feature extraction — RGB HOG, always 1568 features ────────────────────────
def extract_features(img: Image.Image):
    img_rgb   = img.convert('RGB').resize(IMAGE_SIZE)
    img_array = np.array(img_rgb)
    return hog(img_array, orientations=8, pixels_per_cell=(8,8), cells_per_block=(2,2), channel_axis=-1)

# ── Load model ─────────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR,'model.pkl'),   'rb') as f: model   = pickle.load(f)
with open(os.path.join(MODEL_DIR,'scaler.pkl'),  'rb') as f: scaler  = pickle.load(f)
with open(os.path.join(MODEL_DIR,'encoder.pkl'), 'rb') as f: encoder = pickle.load(f)
with open(os.path.join(MODEL_DIR,'metrics.json'))      as f: metrics = json.load(f)

# ── Page ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Food Calorie Estimator", page_icon="🍽️")
st.title("🍽️ Food Recognition & Calorie Estimator")
st.caption("Prodigy Infotech — Task 05 · Random Forest + HOG Features")
st.divider()

# ── Stats ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model",       "Random Forest")
col2.metric("Accuracy",    f"{metrics['accuracy']}%")
col3.metric("Food Classes", metrics['n_classes'])
col4.metric("Data",        "Real" if metrics['using_real'] else "Demo")

if not metrics['using_real']:
    st.info("ℹ️ Running in Demo Mode. For accurate food recognition, use the real Kaggle Food-101 dataset.")

st.divider()

# ── Calorie reference table ────────────────────────────────────────────────────
with st.expander("📋 View Calorie Database (all foods)"):
    import pandas as pd
    rows = []
    for key, info in CALORIE_DB.items():
        rows.append({
            'Food': key.replace('_',' ').title(),
            'Calories': info['calories'],
            'Protein (g)': info['protein'],
            'Carbs (g)': info['carbs'],
            'Fat (g)': info['fat'],
            'Serving': info['serving'],
            'Level': get_calorie_color(info['calories'])
        })
    df = pd.DataFrame(rows).sort_values('Food')
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# ── Upload ─────────────────────────────────────────────────────────────────────
st.subheader("📤 Upload a Food Image")
st.write("Upload a photo of any food item to identify it and get its calorie info.")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Food Image", use_container_width=True)
    with col2:
        st.write("**Image Info:**")
        st.write(f"- Size: {img.size[0]} × {img.size[1]} px")
        st.write(f"- Resized to: 64 × 64 px for model")

    st.divider()

    if st.button("🔍 Identify Food & Get Calories", type="primary", use_container_width=True):
        with st.spinner("Analyzing food..."):
            features        = extract_features(img)
            features_scaled = scaler.transform([features])
            prediction      = model.predict(features_scaled)[0]
            proba           = model.predict_proba(features_scaled)[0]
            food_key        = encoder.inverse_transform([prediction])[0]
            food_name       = food_key.replace('_', ' ').title()
            confidence      = proba[prediction] * 100
            info            = get_calorie_info(food_key)
            cal_level       = get_calorie_color(info['calories'])

            st.subheader("🍴 Identified Food")
            st.success(f"### {food_name}")
            st.metric("Confidence", f"{confidence:.1f}%")

            st.divider()
            st.subheader("📊 Nutrition Info")
            st.write(f"**Serving size:** {info['serving']}")
            st.write(f"**Calorie Level:** {cal_level}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔥 Calories", f"{info['calories']} kcal")
            c2.metric("💪 Protein",  f"{info['protein']}g")
            c3.metric("🍞 Carbs",    f"{info['carbs']}g")
            c4.metric("🧈 Fat",      f"{info['fat']}g")

            # Macros bar chart
            st.divider()
            st.write("**Macronutrient Breakdown:**")
            total_macro = info['protein'] + info['carbs'] + info['fat']
            if total_macro > 0:
                import pandas as pd
                macro_df = pd.DataFrame({
                    'Nutrient': ['Protein', 'Carbs', 'Fat'],
                    'Grams':    [info['protein'], info['carbs'], info['fat']]
                })
                st.bar_chart(macro_df.set_index('Nutrient'))

            st.divider()
            st.write("**Top 3 Predictions:**")
            top3 = np.argsort(proba)[::-1][:3]
            for rank, idx in enumerate(top3):
                f_key   = encoder.inverse_transform([idx])[0]
                f_name  = f_key.replace('_',' ').title()
                f_conf  = proba[idx] * 100
                f_info  = get_calorie_info(f_key)
                st.write(f"{rank+1}. **{f_name}** — `{f_conf:.1f}%` · {f_info['calories']} kcal")
                st.progress(float(proba[idx]))

st.divider()
st.subheader("📦 Kaggle Dataset")
st.markdown("[👉 Download Food-101 Dataset](https://www.kaggle.com/dansbecker/food-101)")
st.write("""
**To use real data:**
1. Download and unzip the dataset
2. Place food folders inside `data/train/`  
   Example: `data/train/pizza/`, `data/train/sushi/`, etc.
3. Run `python model/train_model.py`
4. Restart the app
""")
