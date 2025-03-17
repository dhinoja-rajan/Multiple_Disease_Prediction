import os
import streamlit as st
import joblib
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🩺")
    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models
models = {
    "Breast Cancer": joblib.load(f'{working_dir}/saved_models/BREAST_CANCER_Prediction_Model.pkl'),
    "Diabetes": joblib.load(f'{working_dir}/saved_models/DIABETES_Disease_Prediction_Model.pkl'),
    "Heart Disease": joblib.load(f'{working_dir}/saved_models/HEART_Disease_Prediction_Model.pkl'),
    "Kidney Disease": joblib.load(f'{working_dir}/saved_models/KIDNEY_Disease_Prediction_Model.pkl'),
    "Liver Disease": joblib.load(f'{working_dir}/saved_models/LIVER_Disease_Prediction_Model.pkl'),
    "Parkinson's Disease": joblib.load(f'{working_dir}/saved_models/PARKINSON_Disease_Prediction_Model.pkl'),
}

# ✅ Sidebar with tab-like structure
with st.sidebar:
    selected_disease = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Disease Prediction", "Heart Disease Prediction", "Liver Disease Prediction", "Kidney Disease Prediction", "Breast Cancer Prediction", "Parkinson's Disease Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart-fill", "lungs-fill", "droplet-fill", "plus", "person-fill"],
        default_index=0
    )

# ✅ Diabetes Prediction Page
if selected_disease == "Diabetes Disease Prediction":
    st.title("🩺 Diabetes Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.text_input("Number of Pregnancies", "0")
        skin_thickness = st.text_input("Skin Thickness value", "0")

    with col2:
        glucose = st.text_input("Glucose Level", "0")
        insulin = st.text_input("Insulin Level", "0")

    with col3:
        blood_pressure = st.text_input("Blood Pressure value", "0")
        bmi = st.text_input("BMI value", "0")

    diabetes_pedigree = st.text_input("Diabetes Pedigree Function value", "0")
    age = st.text_input("Age", "0")

    if st.button("Diabetes Test Result"):
        user_input = [float(x) for x in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
        prediction = models["Diabetes"].predict([user_input])
        st.success("The person is diabetic" if prediction[0] == 1 else "The person is not diabetic")

# ✅ Heart Disease Prediction Page
if selected_disease == "Heart Disease Prediction":
    st.title("💖 Heart Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age", "0")
        trestbps = st.text_input("Resting Blood Pressure", "0")
        restecg = st.text_input("Resting Electrocardiographic results", "0")
        oldpeak = st.text_input("ST depression induced by exercise", "0")

    with col2:
        sex = st.text_input("Sex (0 = Female, 1 = Male)", "0")
        chol = st.text_input("Serum Cholesterol in mg/dl", "0")
        thalach = st.text_input("Maximum Heart Rate achieved", "0")
        slope = st.text_input("Slope of the peak exercise ST segment", "0")

    with col3:
        cp = st.text_input("Chest Pain types", "0")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", "0")
        exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)", "0")
        ca = st.text_input("Major vessels colored by fluoroscopy", "0")

    thal = st.text_input("Thal (0 = normal, 1 = fixed defect, 2 = reversible defect)", "0")

    if st.button("Heart Disease Test Result"):
        user_input = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        prediction = models["Heart Disease"].predict([user_input])
        st.success("The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease")

# ✅ Parkinson's Prediction Page
if selected_disease == "Parkinson's Disease Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input("MDVP:Fo(Hz)", "0")
        jitter_abs = st.text_input("MDVP:Jitter(Abs)", "0")
        shimmer_dB = st.text_input("MDVP:Shimmer(dB)", "0")
        APQ5 = st.text_input("Shimmer:APQ5", "0")
        spread2 = st.text_input("Spread2", "0")

    with col2:
        fhi = st.text_input("MDVP:Fhi(Hz)", "0")
        RAP = st.text_input("MDVP:RAP", "0")
        APQ = st.text_input("MDVP:APQ", "0")
        DDA = st.text_input("Shimmer:DDA", "0")
        D2 = st.text_input("D2", "0")

    with col3:
        flo = st.text_input("MDVP:Flo(Hz)", "0")
        PPQ = st.text_input("MDVP:PPQ", "0")
        shimmer = st.text_input("MDVP:Shimmer", "0")
        NHR = st.text_input("NHR", "0")
        PPE = st.text_input("PPE", "0")

    with col4:
        jitter_percent = st.text_input("MDVP:Jitter(%)", "0")
        DDP = st.text_input("Jitter:DDP", "0")
        APQ3 = st.text_input("Shimmer:APQ3", "0")
        HNR = st.text_input("HNR", "0")

    with col5:
        RPDE = st.text_input("RPDE", "0")
        DFA = st.text_input("DFA", "0")
        spread1 = st.text_input("Spread1", "0")

    if st.button("Parkinson's Test Result"):
        user_input = [float(x) for x in [fo, fhi, flo, jitter_percent, jitter_abs, RAP, PPQ, DDP, shimmer, shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
        prediction = models["Parkinson's"].predict([user_input])
        st.success("The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease")

# ✅ Liver Disease Prediction Page
if selected_disease == "Liver Disease Prediction":
    st.title("🩺 Liver Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age", "0")
        total_bilirubin = st.text_input("Total Bilirubin", "0")
        total_proteins = st.text_input("Total Proteins", "0")

    with col2:
        direct_bilirubin = st.text_input("Direct Bilirubin", "0")
        albumin = st.text_input("Albumin", "0")
        albumin_globulin_ratio = st.text_input("Albumin and Globulin Ratio", "0")

    with col3:
        alkaline_phosphatase = st.text_input("Alkaline Phosphatase", "0")
        sgpt = st.text_input("SGPT (Alanine Aminotransferase)", "0")
        sgot = st.text_input("SGOT (Aspartate Aminotransferase)", "0")

    gender = st.text_input("Gender (0 = Female, 1 = Male)", "0")

    if st.button("Liver Disease Test Result"):
        user_input = [float(x) for x in [age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphatase, sgpt, sgot, total_proteins, albumin, albumin_globulin_ratio]]
        prediction = models["Liver Disease"].predict([user_input])
        st.success("The person has Liver Disease" if prediction[0] == 1 else "The person does not have Liver Disease")

# ✅ Kidney Disease Prediction Page
if selected_disease == "Kidney Disease Prediction":
    st.title("🩺 Kidney Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age", "0")
        blood_pressure = st.text_input("Blood Pressure", "0")
        specific_gravity = st.text_input("Specific Gravity", "0")

    with col2:
        albumin = st.text_input("Albumin", "0")
        sugar = st.text_input("Sugar Level", "0")
        red_blood_cells = st.text_input("Red Blood Cells (0 = Normal, 1 = Abnormal)", "0")

    with col3:
        pus_cells = st.text_input("Pus Cells (0 = Normal, 1 = Abnormal)", "0")
        hemoglobin = st.text_input("Hemoglobin", "0")
        serum_creatinine = st.text_input("Serum Creatinine", "0")

    sodium = st.text_input("Sodium Level", "0")

    if st.button("Kidney Disease Test Result"):
        user_input = [float(x) for x in [age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cells, hemoglobin, serum_creatinine, sodium]]
        prediction = models["Kidney Disease"].predict([user_input])
        st.success("The person has Kidney Disease" if prediction[0] == 1 else "The person does not have Kidney Disease")

# ✅ Breast Cancer Prediction Page
if selected_disease == "Breast Cancer Prediction":
    st.title("🎗️ Breast Cancer Prediction using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.text_input("Mean Radius", "0")
        mean_texture = st.text_input("Mean Texture", "0")
        mean_smoothness = st.text_input("Mean Smoothness", "0")

    with col2:
        mean_perimeter = st.text_input("Mean Perimeter", "0")
        mean_compactness = st.text_input("Mean Compactness", "0")
        mean_concave_points = st.text_input("Mean Concave Points", "0")

    with col3:
        mean_area = st.text_input("Mean Area", "0")
        mean_symmetry = st.text_input("Mean Symmetry", "0")
        mean_fractal_dimension = st.text_input("Mean Fractal Dimension", "0")

    if st.button("Breast Cancer Test Result"):
        user_input = [float(x) for x in [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concave_points, mean_symmetry, mean_fractal_dimension]]
        prediction = models["Breast Cancer"].predict([user_input])
        st.success("The person has Breast Cancer" if prediction[0] == 1 else "The person does not have Breast Cancer")

