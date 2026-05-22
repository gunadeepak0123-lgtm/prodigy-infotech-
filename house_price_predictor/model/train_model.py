"""
Train a Linear Regression model on house price data.
Uses synthetic data based on Kaggle House Prices dataset structure.
Replace with actual Kaggle data by placing train.csv in this folder.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

def generate_synthetic_data(n=1000, seed=42):
    """Generate realistic synthetic house price data."""
    np.random.seed(seed)
    sqft = np.random.randint(500, 6000, n)
    bedrooms = np.random.randint(1, 7, n)
    bathrooms = np.random.randint(1, 5, n)

    # Realistic price formula with noise
    price = (
        50000
        + sqft * 120
        + bedrooms * 15000
        + bathrooms * 25000
        + np.random.normal(0, 30000, n)
    )
    price = np.clip(price, 50000, 2000000)

    return pd.DataFrame({
        'GrLivArea': sqft,
        'BedroomAbvGr': bedrooms,
        'FullBath': bathrooms,
        'SalePrice': price
    })


def train(data_path=None):
    # Load real data if available
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
        df = df[features + ['SalePrice']].dropna()
    else:
        print("No dataset found. Using synthetic data.")
        df = generate_synthetic_data()

    X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']].values
    y = df['SalePrice'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : ${rmse:,.0f}")

    # Save model artifacts
    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics
    metrics = {
        'r2': round(r2, 4),
        'rmse': round(rmse, 2),
        'n_samples': len(df),
        'coefficients': {
            'sqft': round(model.coef_[0], 4),
            'bedrooms': round(model.coef_[1], 4),
            'bathrooms': round(model.coef_[2], 4),
        },
        'intercept': round(model.intercept_, 2)
    }
    import json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print("Model saved successfully.")
    return metrics


if __name__ == '__main__':
    kaggle_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
    train(data_path=kaggle_path)
