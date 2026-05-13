import streamlit as st
import joblib
import numpy as np

# Page config (recommended fix)
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load trained model (safe loading)
model = joblib.load("house_price_model.pkl")

# Title
st.title("🏠 House Price Prediction System")

st.write(
    "This machine learning app predicts house prices "
    "using a Random Forest Regression model."
)

st.header("Enter House Features")

# Inputs
overallqual = st.slider("Overall Quality", 1, 10, 5)

grlivarea = st.number_input("Living Area (sq ft)", 100, 10000, 1500)

garagecars = st.number_input("Garage Capacity", 0, 5, 2)

totalbsmtsf = st.number_input("Basement Area (sq ft)", 0, 5000, 800)

fullbath = st.number_input("Number of Bathrooms", 0, 10, 2)

# Prediction
if st.button("Predict House Price"):

    try:
        input_data = np.array([[overallqual, grlivarea, garagecars, totalbsmtsf, fullbath]])

        prediction = model.predict(input_data)

        st.success(f"🏡 Estimated House Price: ₱{prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.write("---")
st.write("Artificial Intelligence 5.0")
st.write("Machine Learning Performance Evaluation")