import pandas as pd
import numpy as np

FEATURES = ["total_sqft", "bath", "balcony"]

TARGET = "price"

def load_data(path):

    df = pd.read_csv(path)

    df = df[FEATURES + [TARGET]]

    df = df.dropna()

    return df

def normalize(X):

    mean = np.mean(X, axis=0)

    std = np.std(X, axis=0)

    X_norm = (X - mean) / std

    return X_norm, mean, std

def split_data(X, y, train_ratio=0.8):

    n = len(X)

    split_idx = int(n * train_ratio)

    return (
        X[:split_idx],
        X[split_idx:],
        y[:split_idx],
        y[split_idx:]
    )

def convert_sqft_to_num(x):

    x = str(x).strip()

    try:
        return float(x)
    
    except ValueError:

        if "-" in str(x):

            low, high = x.split("-")

            return (float(low) + float(high) / 2)
    
        return None



