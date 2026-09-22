import numpy as np
import json

from preprocess import *
from model import *
from metrics import *

df = load_data("../data/Bengaluru_House_Data.csv")

location_stats = df["location"].value_counts()

location_stats_less_than_10 = location_stats[location_stats <= 10]

df["location"] = df["location"].apply(lambda x: "other" if x in location_stats_less_than_10 else x)

location_dummies = pd.get_dummies(df["location"], dtype=int)

df = pd.concat([df, location_dummies], axis = 1)

df.drop("location", axis=1, inplace=True)

df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
df["size"] = df["size"].apply(extract_no_of_bedrooms)

df = df.dropna(subset=["total_sqft"])

FEATURES = df.drop("price", axis=1).columns.tolist()

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
    "std": std.tolist(),
    "y_mean": float(mean_y),
    "y_std": float(std_y),
    "columns": FEATURES
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

print("\nSample Prediction\n")

y_test_actual = y_test * std_y + mean_y
predictions_actual = predictions * std_y + mean_y

for i in range(10):
    error = abs(y_test_actual[i] - predictions_actual[i])
    percent = (error / y_test_actual[i]) * 100
    print(f"Actual: {y_test_actual[i]:.2f} | "f"Predicted: {predictions_actual[i]:.2f} |" f"Error: {error}")