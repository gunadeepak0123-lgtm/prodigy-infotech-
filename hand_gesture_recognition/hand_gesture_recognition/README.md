# 🤚 Hand Gesture Recognition
**Prodigy Infotech ML Internship · Task-04**

Recognizes hand gestures from images using HOG features + Random Forest classifier.

---

## 📁 Project Structure

```
hand_gesture_recognition/
├── streamlit_app.py       ← Main app (pure Python)
├── requirements.txt       ← Dependencies
├── README.md
├── model/
│   ├── train_model.py     ← Training script
│   ├── model.pkl          ← Trained model
│   ├── scaler.pkl         ← StandardScaler
│   ├── encoder.pkl        ← Label encoder
│   └── metrics.json       ← Accuracy & gesture list
└── data/
    └── train/
        ├── 01_palm/       ← Palm gesture images
        ├── 02_l/          ← L gesture images
        ├── 03_fist/       ← Fist gesture images
        └── ...            ← Other gesture folders
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at: **http://localhost:8501**

---

## 📊 Using Real Kaggle Data

Download from:
> https://www.kaggle.com/gti-upm/leapgestrecog

Place gesture folders inside `data/train/` then retrain:

```bash
python model/train_model.py
```

---

## 🖐️ Supported Gestures (10 classes)

| Folder | Gesture |
|--------|---------|
| 01_palm | Palm 🖐️ |
| 02_l | L 🤙 |
| 03_fist | Fist ✊ |
| 04_fist_moved | Fist Moved ✊ |
| 05_thumb | Thumb Up 👍 |
| 06_index | Index ☝️ |
| 07_ok | OK 👌 |
| 08_palm_moved | Palm Moved 🖐️ |
| 09_c | C 🤏 |
| 10_down | Down 👇 |

---

## 🧠 How It Works

1. Image → grayscale → resized to 64×64
2. **HOG features** extracted (shape & edge direction)
3. **StandardScaler** normalizes features
4. **Random Forest** (100 trees) predicts gesture
5. Top 3 predictions shown with confidence %
