import os, pickle, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from skimage.feature import hog

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, '..', 'data', 'train')
IMAGE_SIZE = (64, 64)

GESTURE_LABELS = {
    '01_palm': 'Palm', '02_l': 'L', '03_fist': 'Fist',
    '04_fist_moved': 'Fist Moved', '05_thumb': 'Thumb Up',
    '06_index': 'Index', '07_ok': 'OK',
    '08_palm_moved': 'Palm Moved', '09_c': 'C', '10_down': 'Down',
}

SYNTHETIC_GESTURES = ['Palm', 'Fist', 'Thumb Up', 'Index', 'OK', 'Peace', 'C', 'L', 'Down', 'Palm Moved']

def extract_features(img_path):
    img = Image.open(img_path).convert('L').resize(IMAGE_SIZE)
    arr = np.array(img)
    return hog(arr, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))

def load_real_data(max_per_class=200):
    X, y = [], []
    if not os.path.exists(DATA_DIR): return X, y, []
    gesture_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    labels_used = []
    for gdir in gesture_dirs:
        label = GESTURE_LABELS.get(gdir, gdir)
        folder = os.path.join(DATA_DIR, gdir)
        image_files = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.jpg','.jpeg','.png')):
                    image_files.append(os.path.join(root, f))
        image_files = image_files[:max_per_class]
        for img_path in image_files:
            try:
                X.append(extract_features(img_path))
                y.append(label)
            except: continue
        if image_files:
            labels_used.append(label)
    return X, y, labels_used

def generate_synthetic():
    # Use correct HOG feature size: 1764
    np.random.seed(42)
    n_features = 1764
    X, y = [], []
    for i, gesture in enumerate(SYNTHETIC_GESTURES):
        center = np.zeros(n_features)
        center[i*176:(i+1)*176] = 2.0
        features = np.random.randn(50, n_features) * 0.5 + center
        X.append(features)
        y.extend([gesture] * 50)
    return np.vstack(X), y, SYNTHETIC_GESTURES

def train():
    using_real = False
    X_list, y_list, labels = load_real_data()
    if len(X_list) > 50:
        X = np.array(X_list)
        y = y_list
        using_real = True
        print(f"Loaded {len(X)} real images")
    else:
        print("No images found. Using synthetic data (HOG size: 1764).")
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

    with open(os.path.join(BASE_DIR,'model.pkl'),   'wb') as f: pickle.dump(model, f)
    with open(os.path.join(BASE_DIR,'scaler.pkl'),  'wb') as f: pickle.dump(scaler, f)
    with open(os.path.join(BASE_DIR,'encoder.pkl'), 'wb') as f: pickle.dump(le, f)
    with open(os.path.join(BASE_DIR,'metrics.json'),'w') as f:
        json.dump({'accuracy': round(acc*100,2), 'n_samples': len(X),
                   'n_classes': len(le.classes_), 'gestures': list(le.classes_),
                   'using_real': using_real}, f)
    print("Saved!")

if __name__ == '__main__': train()
