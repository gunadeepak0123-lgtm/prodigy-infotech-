"""
House Price Predictor — Flask Backend
Prodigy Infotech Task-01 | Linear Regression
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

app = Flask(__name__)

# ── Load model ─────────────────────────────────────────────────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        # Auto-train on first run
        import sys
        sys.path.insert(0, MODEL_DIR)
        from train_model import train
        kaggle = os.path.join(BASE_DIR, 'data', 'train.csv')
        train(data_path=kaggle if os.path.exists(kaggle) else None)

    with open(MODEL_PATH, 'rb')  as f: model  = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(METRICS_PATH)      as f: metrics = json.load(f)
    return model, scaler, metrics


model, scaler, metrics = load_artifacts()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sqft      = float(data['sqft'])
        bedrooms  = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])

        # Validation
        if sqft <= 0 or bedrooms <= 0 or bathrooms <= 0:
            return jsonify({'error': 'All values must be positive'}), 400
        if sqft > 20000:
            return jsonify({'error': 'Square footage seems too large (max 20,000)'}), 400

        X = np.array([[sqft, bedrooms, bathrooms]])
        X_scaled = scaler.transform(X)
        price = model.predict(X_scaled)[0]
        price = max(price, 10000)

        # Price range estimate (±15%)
        low  = price * 0.85
        high = price * 1.15

        # Feature contributions (rough breakdown)
        base = model.intercept_
        coefs = model.coef_
        X_s = X_scaled[0]
        contributions = {
            'sqft':      round(abs(coefs[0] * X_s[0])),
            'bedrooms':  round(abs(coefs[1] * X_s[1])),
            'bathrooms': round(abs(coefs[2] * X_s[2])),
        }

        return jsonify({
            'price':         round(price, 2),
            'low':           round(low, 2),
            'high':          round(high, 2),
            'contributions': contributions,
        })

    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400


@app.route('/metrics')
def get_metrics():
    return jsonify(metrics)


# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🏠  House Price Predictor running at http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
