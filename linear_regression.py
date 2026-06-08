import numpy as np
import matplotlib.pyplot as plt
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


def generate_data(n_samples=100, w_true=3.0, b_true=5.0, noise=1.0, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples)
    y = w_true * X + b_true + np.random.randn(n_samples) * noise
    return X, y

def plot_loss_curve(loss_history):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, color='#E07B54', linewidth=2, label='Cost J(w, b)')
    ax.set_xlabel('Epoch / Iteration', fontsize=12)
    ax.set_ylabel('Cost J(w, b) (MSE / 2)', fontsize=12)
    ax.set_title('Cost Function Convergence Curve', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig            