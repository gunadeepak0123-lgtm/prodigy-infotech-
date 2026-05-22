# 🏠 HouseIQ — House Price Predictor
**Prodigy Infotech ML Internship · Task-01**

Linear Regression model to predict house prices based on:
- Square Footage (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Bathrooms (FullBath)

---

## 📁 Project Structure

```
house_price_predictor/
├── app.py                  ← Flask web server
├── requirements.txt        ← Python dependencies
├── README.md
├── model/
│   ├── train_model.py      ← Training script
│   ├── model.pkl           ← Trained model (auto-generated)
│   ├── scaler.pkl          ← StandardScaler (auto-generated)
│   └── metrics.json        ← Model performance metrics
├── templates/
│   └── index.html          ← Beautiful frontend UI
└── data/                   ← (Optional) Place Kaggle train.csv here
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Use real Kaggle data
Download the dataset from:
> https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Place `train.csv` inside a `data/` folder:
```
house_price_predictor/data/train.csv
```

### 3. Train the model (optional — auto-trains on first run)
```bash
python model/train_model.py
```

### 4. Run the web app
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🧠 Model Details

| Metric | Value |
|--------|-------|
| Algorithm | Linear Regression (scikit-learn) |
| Preprocessing | StandardScaler |
| Features | GrLivArea, BedroomAbvGr, FullBath |
| R² Score | ~97.7% (synthetic) |
| RMSE | ~$29,688 |

---

## 📊 API Endpoint

**POST** `/predict`
```json
{
  "sqft": 2000,
  "bedrooms": 3,
  "bathrooms": 2
}
```

Response:
```json
{
  "price": 285000.0,
  "low": 242250.0,
  "high": 327750.0,
  "contributions": {
    "sqft": 180000,
    "bedrooms": 45000,
    "bathrooms": 60000
  }
}
```
