import streamlit as st
import numpy as np
import joblib

# Load model file
model = joblib.load('model.pkl')

# App title
st.title('Taxi Fare Prediction')

# User inputs
passenger = st.select_slider(
    "Total number of passengers",
    options=[1, 2, 3, 4, 5, 6],
    value=1  # Default single value instead of range
)
distance = st.number_input("Enter the traveling distance (in km)", min_value=0.0, step=0.1)

# Select a tip amount (fixed values instead of range)
extra = st.select_slider(
    "Raise your fare",
    options=[0, 5, 10, 15, 20, 25, 30],
    value=0  # Default single value instead of range
)

# Creating input object
input_value = np.array([[passenger, distance, extra]])

# Predict using the model
try:
    pred = model.predict(input_value)[0] if model.predict(input_value) else 0
except Exception as e:
    pred = 0  # Default to zero if an error occurs # Extracting single prediction value

# Display selected inputs
st.write(f"Total number of passengers (0 - 6): {passenger}")
st.write(f"Total traveling distance: {distance} km")
st.write(f"Raise your fare: {extra}")
st.write(f"### Total Fare (Inc all charges): Rs.{pred:.2f}")
