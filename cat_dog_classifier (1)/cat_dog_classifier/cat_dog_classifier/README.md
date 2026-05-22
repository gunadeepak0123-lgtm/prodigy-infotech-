# 🐾 Cat vs Dog Classifier — SVM
**Prodigy Infotech ML Internship · Task-03**

Classifies images of cats and dogs using a Support Vector Machine (SVM) with HOG features.

---

## 📁 Project Structure

```
cat_dog_classifier/
├── streamlit_app.py       ← Main app (pure Python)
├── requirements.txt       ← Dependencies
├── README.md
├── model/
│   ├── train_model.py     ← Training script
│   ├── svm_model.pkl      ← Trained SVM model
│   ├── scaler.pkl         ← StandardScaler
│   └── metrics.json       ← Accuracy metrics
└── data/
    └── train/
        ├── cats/          ← Place cat images here
        └── dogs/          ← Place dog images here
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
> https://www.kaggle.com/c/dogs-vs-cats/data

Place images in:
- `data/train/cats/` → cat images
- `data/train/dogs/` → dog images

Then retrain:
```bash
python model/train_model.py
```

---

## 🧠 How It Works

1. Image is resized to **64×64 px**
2. **HOG features** extracted (captures shape & texture)
3. Features scaled with **StandardScaler**
4. **SVM with RBF kernel** classifies as Cat or Dog
5. Probability score shows model confidence
