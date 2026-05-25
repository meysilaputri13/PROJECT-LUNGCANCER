"""
generate_data.py
Script untuk generate dataset kanker paru sintetis.
"""

import pandas as pd
import numpy as np
import os

def generate_dataset():
    np.random.seed(42)
    n = 500

    print("🔄 Generating dataset...")

    data = {}

    data['GENDER'] = np.random.choice(['M', 'F'], n)
    data['AGE'] = np.random.randint(21, 85, n)
    data['SMOKING'] = np.random.choice([1, 2], n, p=[0.35, 0.65])
    data['YELLOW_FINGERS'] = np.random.choice([1, 2], n, p=[0.50, 0.50])
    data['ANXIETY'] = np.random.choice([1, 2], n, p=[0.55, 0.45])
    data['PEER_PRESSURE'] = np.random.choice([1, 2], n, p=[0.60, 0.40])
    data['CHRONIC_DISEASE'] = np.random.choice([1, 2], n, p=[0.65, 0.35])
    data['FATIGUE'] = np.random.choice([1, 2], n, p=[0.45, 0.55])
    data['ALLERGY'] = np.random.choice([1, 2], n, p=[0.55, 0.45])
    data['WHEEZING'] = np.random.choice([1, 2], n, p=[0.45, 0.55])
    data['ALCOHOL_CONSUMING'] = np.random.choice([1, 2], n, p=[0.55, 0.45])
    data['COUGHING'] = np.random.choice([1, 2], n, p=[0.40, 0.60])
    data['SHORTNESS_OF_BREATH'] = np.random.choice([1, 2], n, p=[0.45, 0.55])
    data['SWALLOWING_DIFFICULTY'] = np.random.choice([1, 2], n, p=[0.70, 0.30])
    data['CHEST_PAIN'] = np.random.choice([1, 2], n, p=[0.50, 0.50])

    risk_score = (
        (data['SMOKING'] == 2).astype(int) * 3.0 +
        (data['AGE'] > 60).astype(int) * 2.0 +
        (data['ALCOHOL_CONSUMING'] == 2).astype(int) * 1.5 +
        (data['CHRONIC_DISEASE'] == 2).astype(int) * 1.5 +
        (data['WHEEZING'] == 2).astype(int) * 1.5 +
        (data['COUGHING'] == 2).astype(int) * 1.5 +
        (data['SHORTNESS_OF_BREATH'] == 2).astype(int) * 1.0 +
        (data['CHEST_PAIN'] == 2).astype(int) * 1.0 +
        (data['YELLOW_FINGERS'] == 2).astype(int) * 0.8 +
        (data['FATIGUE'] == 2).astype(int) * 0.5 +
        (data['SWALLOWING_DIFFICULTY'] == 2).astype(int) * 0.5 +
        (data['ANXIETY'] == 2).astype(int) * 0.3 +
        (data['GENDER'] == 'M').astype(int) * 0.5 +
        np.random.normal(0, 1.5, n)
    )

    data['LUNG_CANCER'] = (risk_score > 6).map({True: 'YES', False: 'NO'})

    df = pd.DataFrame(data)
    os.makedirs('dataset', exist_ok=True)
    filepath = os.path.join('dataset', 'lung_cancer.csv')
    df.to_csv(filepath, index=False)

    print(f"\n✅ Dataset berhasil dibuat di: {filepath}")
    return df


if __name__ == "__main__":
    generate_dataset()