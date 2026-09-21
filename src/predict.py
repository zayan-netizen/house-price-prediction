from pathlib import Path
import json
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.json"

with open(MODEL_PATH, "r") as f:

    model_data = json.load(f)

weights = np.array(model_data["weights"], dtype=float)

bias = model_data["bias"]

mean = np.array(model_data["mean"], dtype=float)

std = np.array(model_data["std"], dtype=float)

y_mean = model_data["y_mean"]

y_std = model_data["y_std"]

columns = model_data["columns"]

def predict_price(sqft, no_bath, no_balcony, location):

    X = np.zeros(len(columns))

    X[columns.index("total_sqft")] = sqft
    X[columns.index("bath")] = no_bath
    X[columns.index("balcony")] = no_balcony

    if location in columns:
        X[columns.index(location)] = 1
    else:
        X[columns.index("other")] = 1

    X = (X - mean) / std

    prediction = round((np.dot(X, weights) + bias) * y_std + y_mean, 2)

    return prediction

