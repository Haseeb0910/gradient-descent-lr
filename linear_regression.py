import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.05, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 0.0  # weight
        self.b = 0.0  # bias
        self.w_history = []
        self.b_history = []
        self.loss_history = []

    def predict(self, X):
        return self.w * X + self.b

    def compute_cost(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def compute_gradients(self, X, y):
        m = len(y)
        predictions = self.predict(X)
        errors = predictions - y
        dw = (1 / m) * np.sum(errors * X)
        db = (1 / m) * np.sum(errors)
        return dw, db

    def fit(self, X, y):
        for _ in range(self.epochs):
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            self.loss_history.append(self.compute_cost(X, y))
            dw, db = self.compute_gradients(X, y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db