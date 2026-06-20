import numpy as np

class LinearRegression:

    def __init__(self,
                 learning_rate = 0.001,
                 epochs = 10000
            ):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)

        self.bias = 0

        for epoch in range(self.epochs):

            predictions = (np.dot(X, self.weights) + self.bias)

            cost = np.mean((predictions - y) ** 2)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: {cost}")

            dw = 1 / samples * (np.dot(X.T, (predictions - y)))

            db = 1 / samples * (np.sum(predictions - y))

            self.weights -= self.lr * dw

            self.bias -= self.lr * db
        
    def predict(self, X):

        return(
            np.dot(X, self.weights) + self.bias
        )