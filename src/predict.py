import json
import numpy as np

with open("../models/model.json", "r") as f:

    model_data = json.load(f)

weights = np.array(model_data["weights"])
bias = model_data["bias"]
mean = np.array(model_data["mean"])
std = np.array(model_data["std"])

def predict_price(sqft, bhk, year):

    X = np.array([sqft, bhk, year])

    X = (X - mean) / std

    prediction = (np.dot(X, weights) + bias)

    return prediction