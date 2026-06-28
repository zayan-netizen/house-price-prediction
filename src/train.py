import numpy as np
import json

from preprocess import *
from model import *
from metrics import *

print(f"rows before: {len(df)}")

df = load_data("../data/Bengaluru_House_Data.csv")

df["total_sqft"] = (df["total_sqft"].apply(convert_sqft_to_num))

df = df.dropna(subset=["total_sqft"])

print(f"rows after: {len(df)}")

X = df[FEATURES].values

y = df[TARGET].values

X, mean, std = normalize(X)

y, mean_y, std_y = normalize(y)
 
(X_train, X_test, y_train, y_test) = split_data(X, y)

model = LinearRegression(learning_rate=0.01, epochs=5000)

model.fit(X_train, y_train)

print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")

model_data = {
    "weights": model.weights.tolist(),
    "bias": float(model.bias),
    "mean": mean.tolist(),
    "std": std.tolist()
}

with open("../models/model.json", "w") as f:

    json.dump(model_data, f, indent=4)

    print("Model saved")

predictions = model.predict(X_test)

print(
    "MAE:",
    mae(y_test, predictions)
)

print(
    "RMSE:",
    rmse(y_test, predictions)
)

print(
    "R²:",
    r2_score(y_test, predictions)
)

