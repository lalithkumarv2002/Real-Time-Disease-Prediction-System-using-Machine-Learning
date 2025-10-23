import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("disease_model.joblib")

st.title("🏥 Real-Time Disease Prediction System")
st.write("This tool predicts possible diseases based on common symptoms. Works 100% offline.")

# Input fields
fever = st.checkbox("Do you have fever?")
cough = st.checkbox("Do you have cough?")
fatigue = st.checkbox("Are you feeling fatigue?")
headache = st.checkbox("Do you have a headache?")
body_pain = st.checkbox("Do you have body pain?")

# Convert inputs to model format
input_data = np.array([[fever, cough, fatigue, headache, body_pain]]).astype(int)

if st.button("Predict Disease"):
    prediction = model.predict(input_data)[0]
    st.success(f"🩺 Predicted Disease: **{prediction}**")
