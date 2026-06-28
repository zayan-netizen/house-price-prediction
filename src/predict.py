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

def predict_price(sqft, no_bath, no_balcony):

    X = np.array([sqft, no_bath, no_balcony], dtype=float)

    X = (X - mean) / std

    prediction = (np.dot(X, weights) + bias) * y_std + y_mean

    return prediction

