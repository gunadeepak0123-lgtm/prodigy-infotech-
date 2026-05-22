"""
Train a Random Forest model for food recognition.
Uses HOG features from images.

Kaggle dataset: https://www.kaggle.com/dansbecker/food-101
- Place food folders inside data/train/
- Each folder = one food category (apple_pie, pizza, sushi, etc.)

If no dataset found, uses synthetic data for demo.
"""

import os, pickle, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from skimage.feature import hog

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data', 'train')
IMAGE_SIZE = (64, 64)

# Calorie database (per 100g / per serving) for common Food-101 items
CALORIE_DB = {
    'apple_pie':           {'calories': 237, 'protein': 2, 'carbs': 34, 'fat': 11, 'serving': '1 slice (125g)'},
    'baby_back_ribs':      {'calories': 297, 'protein': 26,'carbs': 0,  'fat': 21, 'serving': '100g'},
    'baklava':             {'calories': 428, 'protein': 6, 'carbs': 52, 'fat': 23, 'serving': '1 piece (85g)'},
    'beef_carpaccio':      {'calories': 150, 'protein': 20,'carbs': 1,  'fat': 8,  'serving': '100g'},
    'bibimbap':            {'calories': 490, 'protein': 26,'carbs': 69, 'fat': 12, 'serving': '1 bowl'},
    'bread_pudding':       {'calories': 153, 'protein': 5, 'carbs': 25, 'fat': 4,  'serving': '100g'},
    'breakfast_burrito':   {'calories': 305, 'protein': 14,'carbs': 30, 'fat': 14, 'serving': '1 burrito'},
    'bruschetta':          {'calories': 194, 'protein': 6, 'carbs': 28, 'fat': 7,  'serving': '100g'},
    'caesar_salad':        {'calories': 158, 'protein': 4, 'carbs': 8,  'fat': 13, 'serving': '1 bowl'},
    'cannoli':             {'calories': 225, 'protein': 5, 'carbs': 26, 'fat': 12, 'serving': '1 piece'},
    'caprese_salad':       {'calories': 166, 'protein': 10,'carbs': 4,  'fat': 12, 'serving': '100g'},
    'carrot_cake':         {'calories': 415, 'protein': 4, 'carbs': 55, 'fat': 21, 'serving': '1 slice'},
    'ceviche':             {'calories': 130, 'protein': 18,'carbs': 7,  'fat': 3,  'serving': '100g'},
    'cheesecake':          {'calories': 321, 'protein': 5, 'carbs': 26, 'fat': 23, 'serving': '1 slice'},
    'chicken_curry':       {'calories': 150, 'protein': 12,'carbs': 7,  'fat': 8,  'serving': '100g'},
    'chicken_wings':       {'calories': 290, 'protein': 27,'carbs': 0,  'fat': 19, 'serving': '100g'},
    'chocolate_cake':      {'calories': 371, 'protein': 5, 'carbs': 51, 'fat': 17, 'serving': '1 slice'},
    'club_sandwich':       {'calories': 590, 'protein': 35,'carbs': 40, 'fat': 29, 'serving': '1 sandwich'},
    'crab_cakes':          {'calories': 185, 'protein': 14,'carbs': 8,  'fat': 11, 'serving': '1 cake'},
    'donuts':              {'calories': 452, 'protein': 5, 'carbs': 51, 'fat': 25, 'serving': '1 donut'},
    'dumplings':           {'calories': 336, 'protein': 14,'carbs': 33, 'fat': 17, 'serving': '6 pieces'},
    'edamame':             {'calories': 122, 'protein': 11,'carbs': 10, 'fat': 5,  'serving': '100g'},
    'eggs_benedict':       {'calories': 590, 'protein': 24,'carbs': 30, 'fat': 39, 'serving': '1 serving'},
    'escargots':           {'calories': 130, 'protein': 22,'carbs': 2,  'fat': 4,  'serving': '100g'},
    'falafel':             {'calories': 333, 'protein': 13,'carbs': 32, 'fat': 18, 'serving': '100g'},
    'filet_mignon':        {'calories': 267, 'protein': 26,'carbs': 0,  'fat': 18, 'serving': '100g'},
    'fish_and_chips':      {'calories': 595, 'protein': 32,'carbs': 50, 'fat': 28, 'serving': '1 serving'},
    'foie_gras':           {'calories': 462, 'protein': 11,'carbs': 5,  'fat': 44, 'serving': '100g'},
    'french_fries':        {'calories': 312, 'protein': 4, 'carbs': 41, 'fat': 15, 'serving': '100g'},
    'french_onion_soup':   {'calories': 350, 'protein': 12,'carbs': 30, 'fat': 20, 'serving': '1 bowl'},
    'french_toast':        {'calories': 311, 'protein': 9, 'carbs': 36, 'fat': 14, 'serving': '2 slices'},
    'fried_calamari':      {'calories': 175, 'protein': 11,'carbs': 12, 'fat': 8,  'serving': '100g'},
    'fried_rice':          {'calories': 163, 'protein': 4, 'carbs': 29, 'fat': 3,  'serving': '100g'},
    'frozen_yogurt':       {'calories': 159, 'protein': 4, 'carbs': 34, 'fat': 2,  'serving': '100g'},
    'garlic_bread':        {'calories': 350, 'protein': 8, 'carbs': 42, 'fat': 17, 'serving': '2 slices'},
    'gnocchi':             {'calories': 130, 'protein': 4, 'carbs': 28, 'fat': 1,  'serving': '100g'},
    'greek_salad':         {'calories': 211, 'protein': 6, 'carbs': 8,  'fat': 17, 'serving': '1 bowl'},
    'grilled_cheese_sandwich': {'calories': 425, 'protein': 20,'carbs': 30,'fat': 25,'serving': '1 sandwich'},
    'grilled_salmon':      {'calories': 206, 'protein': 29,'carbs': 0,  'fat': 9,  'serving': '100g'},
    'guacamole':           {'calories': 150, 'protein': 2, 'carbs': 8,  'fat': 13, 'serving': '100g'},
    'gyoza':               {'calories': 196, 'protein': 10,'carbs': 18, 'fat': 9,  'serving': '5 pieces'},
    'hamburger':           {'calories': 540, 'protein': 34,'carbs': 40, 'fat': 26, 'serving': '1 burger'},
    'hot_and_sour_soup':   {'calories': 91,  'protein': 7, 'carbs': 8,  'fat': 3,  'serving': '1 bowl'},
    'hot_dog':             {'calories': 290, 'protein': 11,'carbs': 24, 'fat': 17, 'serving': '1 hot dog'},
    'huevos_rancheros':    {'calories': 413, 'protein': 20,'carbs': 32, 'fat': 23, 'serving': '1 serving'},
    'hummus':              {'calories': 177, 'protein': 8, 'carbs': 20, 'fat': 9,  'serving': '100g'},
    'ice_cream':           {'calories': 207, 'protein': 4, 'carbs': 24, 'fat': 11, 'serving': '1 scoop'},
    'lasagna':             {'calories': 135, 'protein': 8, 'carbs': 13, 'fat': 5,  'serving': '100g'},
    'lobster_bisque':      {'calories': 180, 'protein': 9, 'carbs': 12, 'fat': 11, 'serving': '1 bowl'},
    'lobster_roll_sandwich':{'calories': 436,'protein': 28,'carbs': 34, 'fat': 20, 'serving': '1 roll'},
    'macaroni_and_cheese': {'calories': 164, 'protein': 6, 'carbs': 22, 'fat': 5,  'serving': '100g'},
    'macarons':            {'calories': 404, 'protein': 5, 'carbs': 64, 'fat': 14, 'serving': '3 pieces'},
    'miso_soup':           {'calories': 40,  'protein': 3, 'carbs': 4,  'fat': 1,  'serving': '1 bowl'},
    'mussels':             {'calories': 86,  'protein': 12,'carbs': 4,  'fat': 2,  'serving': '100g'},
    'nachos':              {'calories': 346, 'protein': 9, 'carbs': 36, 'fat': 19, 'serving': '100g'},
    'omelette':            {'calories': 154, 'protein': 11,'carbs': 1,  'fat': 12, 'serving': '1 omelette'},
    'onion_rings':         {'calories': 411, 'protein': 6, 'carbs': 43, 'fat': 24, 'serving': '100g'},
    'oysters':             {'calories': 68,  'protein': 7, 'carbs': 4,  'fat': 2,  'serving': '6 oysters'},
    'pad_thai':            {'calories': 225, 'protein': 12,'carbs': 32, 'fat': 6,  'serving': '100g'},
    'paella':              {'calories': 190, 'protein': 15,'carbs': 22, 'fat': 5,  'serving': '100g'},
    'pancakes':            {'calories': 227, 'protein': 6, 'carbs': 30, 'fat': 9,  'serving': '2 pancakes'},
    'panna_cotta':         {'calories': 196, 'protein': 3, 'carbs': 22, 'fat': 11, 'serving': '100g'},
    'peking_duck':         {'calories': 337, 'protein': 19,'carbs': 7,  'fat': 27, 'serving': '100g'},
    'pho':                 {'calories': 215, 'protein': 15,'carbs': 28, 'fat': 5,  'serving': '1 bowl'},
    'pizza':               {'calories': 266, 'protein': 11,'carbs': 33, 'fat': 10, 'serving': '1 slice'},
    'pork_chop':           {'calories': 231, 'protein': 25,'carbs': 0,  'fat': 14, 'serving': '100g'},
    'poutine':             {'calories': 740, 'protein': 23,'carbs': 90, 'fat': 36, 'serving': '1 serving'},
    'prime_rib':           {'calories': 340, 'protein': 29,'carbs': 0,  'fat': 24, 'serving': '100g'},
    'pulled_pork_sandwich':{'calories': 480, 'protein': 30,'carbs': 38, 'fat': 20, 'serving': '1 sandwich'},
    'ramen':               {'calories': 436, 'protein': 21,'carbs': 57, 'fat': 14, 'serving': '1 bowl'},
    'ravioli':             {'calories': 220, 'protein': 10,'carbs': 30, 'fat': 7,  'serving': '100g'},
    'red_velvet_cake':     {'calories': 367, 'protein': 4, 'carbs': 51, 'fat': 17, 'serving': '1 slice'},
    'risotto':             {'calories': 166, 'protein': 4, 'carbs': 28, 'fat': 4,  'serving': '100g'},
    'samosa':              {'calories': 262, 'protein': 5, 'carbs': 28, 'fat': 15, 'serving': '2 pieces'},
    'sashimi':             {'calories': 127, 'protein': 20,'carbs': 0,  'fat': 5,  'serving': '100g'},
    'scallops':            {'calories': 111, 'protein': 20,'carbs': 5,  'fat': 1,  'serving': '100g'},
    'seaweed_salad':       {'calories': 70,  'protein': 1, 'carbs': 11, 'fat': 2,  'serving': '100g'},
    'shrimp_and_grits':    {'calories': 385, 'protein': 22,'carbs': 34, 'fat': 17, 'serving': '1 serving'},
    'spaghetti_bolognese': {'calories': 300, 'protein': 15,'carbs': 38, 'fat': 9,  'serving': '1 bowl'},
    'spaghetti_carbonara': {'calories': 370, 'protein': 17,'carbs': 40, 'fat': 16, 'serving': '1 bowl'},
    'spring_rolls':        {'calories': 154, 'protein': 4, 'carbs': 18, 'fat': 7,  'serving': '2 rolls'},
    'steak':               {'calories': 271, 'protein': 26,'carbs': 0,  'fat': 18, 'serving': '100g'},
    'strawberry_shortcake':{'calories': 268, 'protein': 4, 'carbs': 36, 'fat': 12, 'serving': '1 slice'},
    'sushi':               {'calories': 143, 'protein': 6, 'carbs': 24, 'fat': 2,  'serving': '6 pieces'},
    'tacos':               {'calories': 226, 'protein': 12,'carbs': 21, 'fat': 10, 'serving': '2 tacos'},
    'takoyaki':            {'calories': 200, 'protein': 10,'carbs': 21, 'fat': 8,  'serving': '6 pieces'},
    'tiramisu':            {'calories': 240, 'protein': 5, 'carbs': 28, 'fat': 12, 'serving': '1 slice'},
    'tuna_tartare':        {'calories': 165, 'protein': 21,'carbs': 2,  'fat': 8,  'serving': '100g'},
    'waffles':             {'calories': 291, 'protein': 8, 'carbs': 37, 'fat': 13, 'serving': '1 waffle'},
}

# Friendly display names
FOOD_NAMES = {k: k.replace('_', ' ').title() for k in CALORIE_DB}

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
    arr = np.array(img)
    return hog(arr, orientations=8, pixels_per_cell=(8,8), cells_per_block=(2,2), channel_axis=-1)

def load_real_data(max_per_class=100):
    X, y = [], []
    if not os.path.exists(DATA_DIR): return X, y, []
    folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    labels_used = []
    for folder in folders:
        path = os.path.join(DATA_DIR, folder)
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))][:max_per_class]
        for fname in files:
            try:
                X.append(extract_features(os.path.join(path, fname)))
                y.append(folder)
            except: continue
        if files: labels_used.append(folder)
    return X, y, labels_used

def generate_synthetic(n_per_class=30, seed=42):
    np.random.seed(seed)
    foods = list(CALORIE_DB.keys())[:20]  # use first 20 foods for demo
    n_features = 1568
    X, y = [], []
    for i, food in enumerate(foods):
        center = np.zeros(n_features)
        center[i*78:(i+1)*78] = 2.0
        features = np.random.randn(n_per_class, n_features) * 0.5 + center
        X.append(features)
        y.extend([food] * n_per_class)
    return np.vstack(X), y, foods

def train():
    using_real = False
    X_list, y_list, labels = load_real_data()
    if len(X_list) > 50:
        X, y = np.array(X_list), y_list
        using_real = True
        print(f"Loaded {len(X)} images across {len(labels)} food classes")
    else:
        print("No images found. Using synthetic data for demo.")
        X, y, labels = generate_synthetic()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"Accuracy: {acc*100:.2f}%")

    with open(os.path.join(BASE_DIR,'model.pkl'),   'wb') as f: pickle.dump(model,  f)
    with open(os.path.join(BASE_DIR,'scaler.pkl'),  'wb') as f: pickle.dump(scaler, f)
    with open(os.path.join(BASE_DIR,'encoder.pkl'), 'wb') as f: pickle.dump(le,     f)
    with open(os.path.join(BASE_DIR,'metrics.json'),'w') as f:
        json.dump({'accuracy': round(acc*100,2), 'n_samples': len(X),
                   'n_classes': len(le.classes_), 'foods': list(le.classes_),
                   'using_real': using_real}, f)
    print("Saved!")

if __name__ == '__main__': train()
