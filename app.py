import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Load the trained model ---
with open('25RP18835.SAV', 'rb') as file:
    model = pickle.load(file)

# --- App Header ---
st.title("🌾 Crop Yield Prediction App")
st.write("Developed by *IMANISHIMWE Nadine*")
st.write("Predict crop yield based on temperature using a trained model.")

# --- User Input ---
st.sidebar.header("Input Features")
temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

# --- Prepare input for prediction ---
input_data = np.array([[temperature]])

# --- Prediction button ---
if st.button("Predict Crop Yield"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🌾 Estimated Crop Yield: **{prediction[0]:.2f} units**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Optional info ---
st.markdown("---")
st.caption("Model file: `25RP18835.SAV` | Developed for local and Streamlit Cloud deployment.")