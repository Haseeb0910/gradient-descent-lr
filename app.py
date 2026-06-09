import streamlit as st
import numpy as np
from linear_regression import (
    LinearRegressionScratch,
    generate_data,
    plot_loss_curve,
    plot_regression_fit,
    plot_contour,
    plot_3d_surface
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Linear Regression From Scratch",
    page_icon="📈",
    layout="wide"
)

# --- TITLE ---
st.title("📈 Linear Regression & Gradient Descent From Scratch")
st.markdown("Adjust the parameters and watch the model learn in real time!")

# --- SIDEBAR SLIDERS ---
st.sidebar.header("⚙️ Model Parameters")

learning_rate = st.sidebar.slider("Learning Rate", 
                    min_value=0.001, max_value=0.5, 
                    value=0.05, step=0.001, format="%.3f")

epochs = st.sidebar.slider("Epochs", 
                    min_value=10, max_value=500, 
                    value=100, step=10)

noise = st.sidebar.slider("Data Noise", 
                    min_value=0.1, max_value=5.0, 
                    value=1.0, step=0.1)

n_samples = st.sidebar.slider("Number of Samples", 
                    min_value=50, max_value=500, 
                    value=100, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("**True Parameters**")
w_true = st.sidebar.slider("True Weight (w)", 
                    min_value=1.0, max_value=10.0, 
                    value=3.0, step=0.5)
b_true = st.sidebar.slider("True Bias (b)", 
                    min_value=1.0, max_value=10.0, 
                    value=5.0, step=0.5)

# --- TRAIN MODEL ---
X, y = generate_data(n_samples=n_samples, w_true=w_true, 
                     b_true=b_true, noise=noise)
model = LinearRegressionScratch(learning_rate=learning_rate, epochs=epochs)
model.fit(X, y)

# --- METRICS ROW ---
st.markdown("### 🎯 Results")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Learned Weight (w)", f"{model.w_history[-1]:.3f}", f"True: {w_true}")
col2.metric("Learned Bias (b)", f"{model.b_history[-1]:.3f}", f"True: {b_true}")
col3.metric("Final Loss", f"{model.loss_history[-1]:.4f}")
col4.metric("Epochs", f"{epochs}")

st.markdown("---")

# --- PLOTS ROW 1 ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📉 Loss Convergence Curve")
    fig1 = plot_loss_curve(model.loss_history)
    st.pyplot(fig1)

with col2:
    st.markdown("### 🎯 Regression Fit Progression")
    fig2 = plot_regression_fit(X, y, model)
    st.pyplot(fig2)

# --- PLOTS ROW 2 ---
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🗺️ 2D Cost Contour Map")
    fig3 = plot_contour(X, y, model)
    st.pyplot(fig3)

with col4:
    st.markdown("### 🏔️ 3D Cost Surface")
    fig4 = plot_3d_surface(X, y, model)
    st.pyplot(fig4)

# --- FOOTER ---
st.markdown("---")
st.markdown("Built from scratch using Python & NumPy | by Haseeb Ur Rehman")