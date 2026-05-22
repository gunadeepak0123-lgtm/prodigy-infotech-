"""
generate_sample_data.py
Generates a sample Mall_Customers-like CSV for testing.
Run: python generate_sample_data.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 200

customer_ids = list(range(1, n + 1))
genders = np.random.choice(['Male', 'Female'], n, p=[0.44, 0.56])
ages = np.random.randint(18, 70, n)
annual_income = np.random.randint(15, 137, n)

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

os.makedirs('data', exist_ok=True)
out = os.path.join('data', 'Mall_Customers_sample.csv')
df.to_csv(out, index=False)
print(f"✓ Saved {len(df)} customers → {out}")
print(df.head())
