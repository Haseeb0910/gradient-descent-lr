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

def plot_regression_fit(X, y, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    X_line = np.linspace(X.min(), X.max(), 100)
    epochs_to_show = [0, 25, 50, 75, model.epochs - 1]
    styles = ['--', ':', ':', ':', '-']
    colors = ['#FF6B6B', '#888888', '#888888', '#FFA500', '#E63946']
    labels = ['Initial Line (Epoch 0)', 'Epoch 25', 'Epoch 50', 'Epoch 75', f'Final Fit (Epoch {model.epochs})']

    ax.scatter(X, y, alpha=0.6, color='#4A90D9', label=f'Data points (y = 3x + 5 + ε)')

    for i, epoch in enumerate(epochs_to_show):
        w = model.w_history[epoch]
        b = model.b_history[epoch]
        y_line = w * X_line + b
        ax.plot(X_line, y_line, linestyle=styles[i], 
                color=colors[i], linewidth=2, label=labels[i])

    ax.set_xlabel('Feature (X)', fontsize=12)
    ax.set_ylabel('Target (y)', fontsize=12)
    ax.set_title('Linear Regression Fit Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig     

def plot_contour(X, y, model):
    w_range = np.linspace(model.w_history[-1] - 4, model.w_history[-1] + 4, 100)
    b_range = np.linspace(model.b_history[-1] - 4, model.b_history[-1] + 4, 100)
    W, B = np.meshgrid(w_range, b_range)

    Z = np.array([
        [(1 / (2 * len(y))) * np.sum((w * X + b - y) ** 2) 
         for w in w_range] 
        for b in b_range
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour_filled = ax.contourf(W, B, Z, levels=30, cmap='plasma', alpha=0.6)
    contour_lines = ax.contour(W, B, Z, levels=30, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour_filled, ax=ax, label='Cost J(w,b)')

    ax.plot(model.w_history, model.b_history, 
            color='#00FFFF', linewidth=2, zorder=5, label='GD Optimization Path')
    
    ax.scatter(model.w_history[0], model.b_history[0], 
               color='yellow', marker='*', s=200, zorder=6, label=f'Start ({model.w_history[0]:.1f}, {model.b_history[0]:.1f})')
    ax.scatter(model.w_history[-1], model.b_history[-1], 
               color='#00FF00', marker='o', s=150, zorder=6, label=f'Fit ({model.w_history[-1]:.2f}, {model.b_history[-1]:.2f})')
    ax.scatter(3.0, 5.0, 
               color='#FF69B4', marker='D', s=150, zorder=6, label='True Optimum (3.0, 5.0)')

    ax.set_xlabel('Weight (w)', fontsize=12)
    ax.set_ylabel('Bias (b)', fontsize=12)
    ax.set_title('2D Cost Contours with Optimization Path', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig