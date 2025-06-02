import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('model.pkl')

# UI
st.title("Iris Flower Prediction App")
st.write("Enter the flower's features to predict the species.")

# Input features
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

# Predict button
if st.button('Predict'):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    st.success(f"The predicted flower is: **{prediction[0]}**")
