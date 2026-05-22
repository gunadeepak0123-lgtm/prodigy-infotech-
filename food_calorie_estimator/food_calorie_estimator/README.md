# 🍽️ Food Recognition & Calorie Estimator
**Prodigy Infotech ML Internship · Task-05**

Recognizes food items from images and estimates their calorie and nutrition content.

---

## 📁 Project Structure

```
food_calorie_estimator/
├── streamlit_app.py       ← Main app (pure Python)
├── requirements.txt       ← Dependencies
├── README.md
├── model/
│   ├── train_model.py     ← Training script
│   ├── model.pkl          ← Trained model
│   ├── scaler.pkl         ← StandardScaler
│   ├── encoder.pkl        ← Label encoder
│   └── metrics.json       ← Metrics
└── data/
    └── train/             ← Place food folders here
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📊 Using Real Kaggle Data

Download from:
> https://www.kaggle.com/dansbecker/food-101

Place food folders inside `data/train/` then:
```bash
python model/train_model.py
```

---

## 🍴 Features

- Identifies food from uploaded image
- Shows calories, protein, carbs, fat
- Macronutrient bar chart
- Top 3 food predictions with confidence
- Full calorie database for 50+ foods
