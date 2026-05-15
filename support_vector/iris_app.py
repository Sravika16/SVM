import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Title
st.title("Iris Flower Prediction App")

st.write("### Iris Dataset")
st.dataframe(df)

# Sidebar Inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

# User Input Data
input_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Train Model
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Prediction
prediction = model.predict(input_data)
prediction_name = iris.target_names[prediction][0]

# Output
st.write("## Prediction")
st.success(f"The predicted flower is: {prediction_name}")