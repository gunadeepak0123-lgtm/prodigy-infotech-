# CustomerIQ — K-Means Customer Segmentation

> **Prodigy InfoTech — Task 02**  
> Group retail customers by purchase behavior using K-Means clustering.

## ✨ Features

- **Interactive K-Means clustering** with adjustable cluster count (K = 2–10)
- **Elbow Method chart** to find the optimal number of clusters
- **Scatter plot visualization** of customer segments
- **Donut chart** showing cluster size distribution
- **Segment profile table** with avg income, spending, age, and gender
- **Upload your own CSV** or use the built-in sample dataset
- **Beautiful dark dashboard UI** built with Flask + Plotly

---

## 📁 Project Structure

```
customer_segmentation/
├── app.py                  # Flask backend + K-Means logic
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   # Uploaded CSVs stored here
├── models/                 # (reserved for saved models)
├── templates/
│   └── index.html          # Main HTML template
└── static/
    ├── css/style.css       # Dashboard styles
    └── js/main.js          # Plotly charts & UI logic
```

---

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
cd customer_segmentation
pip install -r requirements.txt
```

### 2. Run the App

```bash
python app.py
```

### 3. Open in Browser

```
http://localhost:5000
```

---

## 📊 Dataset

The app ships with a built-in sample dataset (200 customers).

You can also use the official Kaggle dataset:  
**https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python**

Download `Mall_Customers.csv` and upload it via the **Upload CSV** button.

### Expected CSV Columns

| Column | Description |
|--------|-------------|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Age in years |
| `Annual Income (k$)` | Annual income in thousands |
| `Spending Score (1-100)` | Spending score assigned by the mall |

Any numeric columns in your CSV can be used as clustering features.

---

## ⚙️ How It Works

1. **Feature Scaling** — StandardScaler normalizes all features to zero mean, unit variance
2. **K-Means++** — Smarter centroid initialization for faster convergence
3. **PCA** — If >2 features selected, Principal Component Analysis reduces to 2D for visualization
4. **Elbow Method** — Inertia plotted for K = 2..10 to help pick the optimal K
5. **Segment Profiles** — Cluster means computed for all selected features

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML | scikit-learn (KMeans, StandardScaler, PCA) |
| Data | pandas, numpy |
| Charts | Plotly.js |
| UI | Vanilla JS, Custom CSS |
| Fonts | Syne (headings), DM Sans (body) |

---

## 📝 Usage Tips

- Start with **K = 5** for the Mall Customers dataset (known optimal)
- Use the **Elbow chart** — look for the "kink" where inertia stops dropping sharply
- Toggle **Age** as a 3rd feature for richer segmentation (uses PCA for 2D viz)
- Hover over scatter points for individual customer details
