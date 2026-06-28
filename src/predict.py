import json
import numpy as np

with open("../models/model.json", "r") as f:

    model_data = json.load(f)

weights = np.array(model_data["weights"], dtype=float)

bias = model_data["bias"]

mean = np.array(model_data["mean"], dtype=float)

std = np.array(model_data["std"], dtype=float)

def predict_price(sqft=1000, bhk=2, year=2000):

    X = np.array([sqft, bhk, year], dtype=float)

    X = (X - mean) / std

    prediction = (np.dot(X, weights) + bias)

    return prediction
