import streamlit as st
import joblib
import numpy as np

# Load models
models = {
    "Breast Cancer": joblib.load("./saved_models/BREAST_CANCER_Prediction_Model.pkl"),
    "Diabetes": joblib.load("./saved_models/DIABETES_Disease_Prediction_Model.pkl"),
    "Heart Disease": joblib.load("./saved_models/HEART_Disease_Prediction_Model.pkl"),
    "Liver Disease": joblib.load("./saved_models/LIVER_Disease_Prediction_Model.pkl"),
    "Kidney Disease": joblib.load("./saved_models/KIDNEY_Disease_Prediction_Model.pkl"),
    "Parkinson's Disease": joblib.load("./saved_models/PARKINSON_Disease_Prediction_Model.pkl"),
}

# Input fields for different diseases
disease_features = {
    "Heart Disease": ["Age", "Sex", "BP", "Cholesterol"],
    "Breast Cancer": ["Radius Mean", "Texture Mean", "Perimeter Mean"],
    "Liver Disease": ["Total Bilirubin", "Alkaline Phosphotase", "SGPT"],
    "Kidney Disease": ["Blood Urea", "Serum Creatinine", "Hemoglobin"],
    "Diabetes": ["Glucose", "Blood Pressure", "BMI"],
    "Parkinson's Disease": ["MDVP: Fo(Hz)", "MDVP: Fhi(Hz)", "MDVP: Flo(Hz)"],
}

# Streamlit App
st.title("Multiple Disease Prediction")

# Select disease
disease = st.selectbox("Select a disease", list(models.keys()))

# Get inputs dynamically
inputs = []
for feature in disease_features[disease]:
    value = st.number_input(f"Enter {feature}", value=0.0)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    model = models[disease]
    prediction = model.predict(np.array(inputs).reshape(1, -1))
    result = "Positive" if prediction[0] == 1 else "Negative"
    st.success(f"Prediction: {result}")
