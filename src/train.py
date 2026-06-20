from preprocess import *
from model import *
from metrics import *

df = load_data("../data/india_housing_prices.csv")

X = df[FEATURES].values

y = df[TARGET].values

X, mean, std = normalize(X)

y, mean_y, std_y = normalize(y)
 
(X_train, X_test, y_train, y_test) = split_data(X, y)

model = LinearRegression(learning_rate=0.01, epochs=2000)

model.fit(X_train, y_train)

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

