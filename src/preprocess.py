import pandas as pd
import numpy as np

FEATURES = ["Size_in_SqFt", "BHK", "Year_Built"]

TARGET = "Price_in_Lakhs"

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

print(normalize(load_data("../data/india_housing_prices.csv")))

def split_data(X, y, train_ratio=0.8):

    n = len(X)

    split_idx = int(n * train_ratio)

    return (
        X[:split_idx],
        X[split_idx:],
        y[:split_idx],
        y[split_idx:]
    )



