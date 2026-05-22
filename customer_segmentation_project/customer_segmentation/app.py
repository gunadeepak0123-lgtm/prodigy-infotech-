from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Sample Dataset Generator ─────────────────────────────────────────────────
def generate_sample_data():
    """Generate realistic retail customer data similar to Mall_Customers.csv"""
    np.random.seed(42)
    n = 200

    customer_ids = list(range(1, n + 1))
    genders = np.random.choice(['Male', 'Female'], n, p=[0.44, 0.56])
    ages = np.random.randint(18, 70, n)
    annual_income = np.random.randint(15, 137, n)  # in k$

    # Spending score with some realistic clusters
    spending = []
    for i in range(n):
        if annual_income[i] < 40:
            spending.append(int(np.clip(np.random.normal(50, 25), 1, 99)))
        elif annual_income[i] < 70:
            spending.append(int(np.clip(np.random.normal(45, 20), 1, 99)))
        else:
            spending.append(int(np.clip(np.random.normal(55, 30), 1, 99)))

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': genders,
        'Age': ages,
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending
    })
    return df


def load_data(filepath=None):
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = generate_sample_data()
    return df


def run_kmeans(df, n_clusters=5, features=None):
    """Run K-Means and return cluster labels + analysis"""
    if features is None:
        features = ['Annual Income (k$)', 'Spending Score (1-100)']

    # Select feature columns that exist
    available = [f for f in features if f in df.columns]
    if not available:
        available = ['Annual Income (k$)', 'Spending Score (1-100)']

    X = df[available].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    # Inertia curve for elbow method (k=2..10)
    inertias = []
    for k in range(2, 11):
        km_tmp = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km_tmp.fit(X_scaled)
        inertias.append(km_tmp.inertia_)

    # PCA for 2D viz if more than 2 features
    if len(available) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        x_label, y_label = 'PC1', 'PC2'
    else:
        coords = X_scaled
        x_label, y_label = available[0], available[1]

    # Build result df
    result_df = df.loc[X.index].copy()
    result_df['Cluster'] = labels

    # Cluster stats
    cluster_stats = []
    for c in range(n_clusters):
        sub = result_df[result_df['Cluster'] == c]
        stat = {'cluster': int(c), 'count': int(len(sub))}
        for col in available:
            stat[f'avg_{col}'] = round(float(sub[col].mean()), 2)
        if 'Age' in sub.columns:
            stat['avg_age'] = round(float(sub['Age'].mean()), 1)
        if 'Gender' in sub.columns and len(sub) > 0:
            stat['dominant_gender'] = sub['Gender'].mode()[0] if len(sub) > 0 else 'N/A'
        cluster_stats.append(stat)

    # Scatter data
    scatter_points = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    for i, (idx, row) in enumerate(result_df.iterrows()):
        scatter_points.append({
            'x': round(float(coords[list(X.index).index(idx)][0]), 4),
            'y': round(float(coords[list(X.index).index(idx)][1]), 4),
            'cluster': int(row['Cluster']),
            'color': colors[int(row['Cluster']) % len(colors)],
            'id': int(row['CustomerID']) if 'CustomerID' in row else i,
            'income': float(row['Annual Income (k$)']) if 'Annual Income (k$)' in row else 0,
            'spending': float(row['Spending Score (1-100)']) if 'Spending Score (1-100)' in row else 0,
            'age': float(row['Age']) if 'Age' in row else 0,
            'gender': str(row['Gender']) if 'Gender' in row else 'N/A',
        })

    return {
        'scatter': scatter_points,
        'cluster_stats': cluster_stats,
        'inertias': inertias,
        'n_clusters': n_clusters,
        'features': available,
        'x_label': x_label,
        'y_label': y_label,
        'total_customers': len(result_df),
        'inertia': round(float(km.inertia_), 2)
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    n_clusters = int(data.get('n_clusters', 5))
    features = data.get('features', ['Annual Income (k$)', 'Spending Score (1-100)'])

    uploaded_path = os.path.join('data', 'uploaded.csv')
    df = load_data(uploaded_path if os.path.exists(uploaded_path) else None)

    result = run_kmeans(df, n_clusters=n_clusters, features=features)
    return jsonify(result)


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    save_path = os.path.join('data', 'uploaded.csv')
    f.save(save_path)
    df = pd.read_csv(save_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return jsonify({
        'columns': df.columns.tolist(),
        'numeric_columns': numeric_cols,
        'rows': len(df),
        'preview': df.head(5).to_dict(orient='records')
    })


@app.route('/api/sample_data')
def sample_data():
    df = generate_sample_data()
    return jsonify({
        'columns': df.columns.tolist(),
        'numeric_columns': ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
        'rows': len(df),
        'preview': df.head(5).to_dict(orient='records')
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
