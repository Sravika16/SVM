import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# -----------------------------
# Title
# -----------------------------
st.title("🌸 Iris Flower Classification using SVC")

# -----------------------------
# Load dataset
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target
class_names = iris.target_names

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train SVC model
# -----------------------------
model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Flower Class"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # scale input
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)

    st.success(f"🌼 Predicted Class: {class_names[prediction[0]]}")