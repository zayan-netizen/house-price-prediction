import pandas as pd
import numpy as np

FEATURES = ["total_sqft", "bath", "balcony", "location", "size"]

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

def extract_no_of_bedrooms(x):

    x = x.strip().split()

    return int(x[0])

'''
# Code for calculating data after removing null values.

df = load_data("../data/Bengaluru_House_Data.csv")

df1 = pd.read_csv("../data/Bengaluru_House_Data.csv")

for item in FEATURES:
    print(f"Difference after removing null values: {item}")
    print(len(df[item].unique()) - len(df1[item].unique()))

print(f"Length without removing null values: {len(df1)}, Length after removing null values: {len(df)}")
'''
