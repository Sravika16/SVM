import streamlit as st
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="California Housing Prediction",
                   layout="wide")

# -----------------------------------
# Title
# -----------------------------------
st.markdown(
    "<h1 style='text-align:center;'>California Housing Price Prediction using SVR</h1>",
    unsafe_allow_html=True
)

# -----------------------------------
# Load Dataset
# -----------------------------------
housing = fetch_california_housing()

X = housing.data
y = housing.target

# -----------------------------------
# Split Data
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Scaling
# -----------------------------------
x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

# -----------------------------------
# Train Model
# -----------------------------------
model = SVR(kernel="rbf")

model.fit(X_train_scaled, y_train_scaled)

# ===================================
# SIDEBAR INPUTS
# ===================================
st.sidebar.header("Enter House Details")

MedInc = st.sidebar.slider(
    "Median Income", 0.0, 15.0, 3.5
)

HouseAge = st.sidebar.slider(
    "House Age", 1.0, 60.0, 25.0
)

AveRooms = st.sidebar.slider(
    "Average Rooms", 1.0, 15.0, 5.0
)

AveBedrms = st.sidebar.slider(
    "Average Bedrooms", 0.5, 5.0, 1.0
)

Population = st.sidebar.slider(
    "Population", 1, 10000, 1000
)

AveOccup = st.sidebar.slider(
    "Average Occupancy", 1.0, 10.0, 3.0
)

Latitude = st.sidebar.slider(
    "Latitude", 32.0, 42.0, 34.0
)

Longitude = st.sidebar.slider(
    "Longitude", -125.0, -114.0, -118.0
)

# ===================================
# CENTER AREA
# ===================================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.markdown("## Predicted House Price")

    if st.button("Predict Price"):

        input_data = np.array([[
            MedInc,
            HouseAge,
            AveRooms,
            AveBedrms,
            Population,
            AveOccup,
            Latitude,
            Longitude
        ]])

        # Scale Input
        input_scaled = x_scaler.transform(input_data)

        # Prediction
        prediction_scaled = model.predict(input_scaled)

        # Convert back
        prediction = y_scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )

        st.success(
            f"${prediction[0][0] * 100000:,.2f}"
        )