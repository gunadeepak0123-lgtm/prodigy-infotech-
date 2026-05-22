import os, pickle, json
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
from skimage.feature import hog

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, '..', 'data', 'train')
IMAGE_SIZE = (64, 64)

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize(IMAGE_SIZE)
    arr = np.array(img)
    return hog(arr, orientations=8, pixels_per_cell=(8,8), cells_per_block=(2,2), channel_axis=-1)

def load_images(folder, label, max_images=500):
    X, y = [], []
    if not os.path.exists(folder): return X, y
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))][:max_images]
    for fname in files:
        try:
            X.append(extract_features(os.path.join(folder, fname)))
            y.append(label)
        except: continue
    return X, y

def generate_synthetic():
    np.random.seed(42)
    # Generate fake HOG-sized features (1568 features)
    sample_size = 1568
    cat = np.random.randn(250, sample_size) + 0.5
    dog = np.random.randn(250, sample_size) - 0.5
    X = np.vstack([cat, dog])
    y = [0]*250 + [1]*250
    return X, y

def train():
    using_real = False
    cat_dir = os.path.join(DATA_DIR, 'cats')
    dog_dir = os.path.join(DATA_DIR, 'dogs')
    Xc, yc = load_images(cat_dir, 0)
    Xd, yd = load_images(dog_dir, 1)
    if len(Xc) > 10 and len(Xd) > 10:
        X = np.array(Xc + Xd)
        y = np.array(yc + yd)
        using_real = True
        print(f"Loaded {len(Xc)} cats, {len(Xd)} dogs")
    else:
        print("No images found. Using synthetic HOG-shaped data.")
        X, y = generate_synthetic()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print(f"Accuracy: {acc*100:.2f}%")

    with open(os.path.join(BASE_DIR,'svm_model.pkl'),'wb') as f: pickle.dump(model,f)
    with open(os.path.join(BASE_DIR,'scaler.pkl'),'wb')    as f: pickle.dump(scaler,f)
    with open(os.path.join(BASE_DIR,'metrics.json'),'w')   as f:
        json.dump({'accuracy':round(acc*100,2),'n_samples':len(X),'using_real':using_real}, f)
    print("Saved!")

if __name__ == '__main__': train()
