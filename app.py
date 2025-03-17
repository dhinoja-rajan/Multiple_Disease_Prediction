import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib


breast_cancer_model = joblib.load("./saved_models/BREAST_CANCER_Prediction_Model.pkl")
breast_cancer_scaler = joblib.load("./saved_models/breast_cancer_scaler.pkl")

diabetes_model = joblib.load("./saved_models/DIABETES_Disease_Prediction_Model.pkl")
diabetes_scaler = joblib.load("./saved_models/diabetes_scaler.pkl")

heart_model = joblib.load("./saved_models/HEART_Disease_Prediction_Model.pkl")
heart_scaler = joblib.load("./saved_models/heart_disease_scaler.pkl")

liver_model = joblib.load("./saved_models/LIVER_Disease_Prediction_Model.pkl")
liver_scaler = joblib.load("./saved_models/liver_disease_scaler.pkl")

parkinson_model = joblib.load("./saved_models/PARKINSON_Disease_Prediction_Model.pkl")
parkinson_scaler = joblib.load("./saved_models/parkinson_scaler.pkl")


# Set page conig
st.set_page_config(
    page_title="Multiple Disease Prediction", layout="wide", page_icon="🧑‍⚕️"
)

# Sidebar or navigation
with st.sidebar:
    selected = option_menu(
        "Multiple Disease Prediction System",
        [
            "Home",
            "Breast Cancer Prediction",
            "Diabetes Disease Prediction",
            "Heart Disease Prediction",
            "Liver Disease Prediction",
            "Parkinson's Disease Prediction",
        ],
        menu_icon="hospital-ill",
        icons=[
            "house-fill",
            "gender-female",
            "activity",
            "heart-fill",
            "droplet-fill",
            "person-fill",
        ],
        default_index=0,
    )

# Routing
if selected == "Home":
    # st.title("Multiple Disease Prediction System")
    # Custom Background (Using Markdown and CSS)
    st.markdown(
        """
        <style>
            body {
                background-color: #F4F4F4;
                font-family: 'Arial', sans-serif;
            }
            .big-title {
                font-size: 40px;
                font-weight: bold;
                color: blue;
                text-align: center;
            }
            .subtext {
                font-size: 18px;
                color: #555;
                text-align: center;
            }
            .quiz-title {
                font-size: 36px;
                font-weight: bold;
                color: violet;
                text-align: center;
            }
            .quiz-subtitle {
                font-size: 18px;
                color: #555;
                text-align: center;
            }
            .question {
                font-size: 22px;
                font-weight: bold;
                margin-top: 20px;
            }
            .option {
                font-size: 18px;
                color: #333;
            }
            .correct {
                color: #4CAF50;
                # font-weight: bold;
            }
            .incorrect {
                color: #FF0000;
                # font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="big-title">🔬 AI-Powered Disease Prediction</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtext">Predict multiple diseases using AI & ML with just a few inputs!</p>',
        unsafe_allow_html=True,
    )

    # Divider Line
    st.markdown("---")

    # Disease Statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🩸 Heart Disease", "17.9M deaths/year", "Leading cause worldwide")
        st.markdown("---")
    with col2:
        st.metric("🎗️ Breast Cancer", "2.3M cases/year", "Most common cancer")
        st.markdown("---")
    with col3:
        st.metric("🦠 Diabetes", "8.5M cases/year", "Worldwide epidemic")
        st.markdown("---")
    with col1:
        st.metric("🩸 Liver Disease", "1.2M cases/year", "Leading cause worldwide")
    with col2:
        st.metric("🧠 Parkinson's Disease", "400K cases/year", "Worldwide epidemic")

    st.markdown("---")

    # Title and subtitle
    st.markdown('<p class="quiz-title">🧠 AI Health Quiz</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="quiz-subtitle">Test your knowledge about health and diseases!</p>',
        unsafe_allow_html=True,
    )

    # Dictionary of questions and answers
    questions = {
        "What is the normal range of blood pressure": {
            "options": ["120/80 mmHg", "150/100 mmHg", "90/60 mmHg"],
            "answer": "120/80 mmHg",
        },
        "Which organ is affected by Parkinson's disease": {
            "options": ["Liver", "Brain", "Heart"],
            "answer": "Brain",
        },
        "Which type of diabetes occurs due to insulin resistance": {
            "options": ["Type 1", "Type 2", "Gestational"],
            "answer": "Type 2",
        },
        "What is the most common type of cancer in women": {
            "options": ["Lung Cancer", "Breast Cancer", "Ovarian Cancer"],
            "answer": "Breast Cancer",
        },
    }

    # Track user responses
    user_answers = {}

    # Loop through each question
    for question, details in questions.items():
        st.markdown(
            f'<span class="question"> {question}❓</span>', unsafe_allow_html=True
        )

        # Add a placeholder option for user to select
        options = ["Select an option"] + details["options"]
        selected_option = st.radio("", options, key=question, index=0)

        if selected_option != "Select an option":
            user_answers[question] = selected_option

        # st.markdown("---")
        st.markdown("<br><br>", unsafe_allow_html=True)

    # Submit Button
    if st.button("Submit Quiz"):
        if len(user_answers) < len(questions):
            st.warning("⚠️ Please answer all questions before submitting!")
        else:
            score = 0
            st.markdown("<hr>", unsafe_allow_html=True)
            for question, details in questions.items():
                if user_answers[question] == details["answer"]:
                    st.markdown(
                        f'<p class="correct">✅ {question} - Correct!</p>',
                        unsafe_allow_html=True,
                    )
                    score += 1
                else:
                    st.markdown(
                        f'<span class="incorrect">❌ {question} - Incorrect.</span> <span style="color: grey;"> The correct answer is "{details["answer"]}".</span>',
                        unsafe_allow_html=True,
                    )

            # Final Score Display
            st.markdown(
                f"<h3 style='text-align: center;'>🎉 Your Score: {score}/{len(questions)} 🎯</h3>",
                unsafe_allow_html=True,
            )

            # Try Again Button
            if st.button("🔄 Try Again"):
                st.experimental_rerun()
    st.markdown("---")

    # Call to Action
    st.markdown("### 🏥 Select a disease from the sidebar to start predicting!")
    st.markdown("---")


elif selected == "Breast Cancer Prediction":
    # breast_cancer_prediction.breast_cancer_prediction()

    st.title("Breast Cancer Prediction Web App")
    st.markdown("### Enter details below to predict breast cancer risk")

    # --- User Inputs ---
    radius_mean = st.number_input(
        "Radius Mean", min_value=5.0, max_value=40.0, step=0.01
    )
    texture_mean = st.number_input(
        "Texture Mean", min_value=5.0, max_value=50.0, step=0.01
    )
    perimeter_mean = st.number_input(
        "Perimeter Mean", min_value=50.0, max_value=200.0, step=0.01
    )
    area_mean = st.number_input(
        "Area Mean", min_value=100.0, max_value=3000.0, step=1.0
    )
    smoothness_mean = st.number_input(
        "Smoothness Mean", min_value=0.01, max_value=1.00, step=0.001
    )
    compactness_mean = st.number_input(
        "Compactness Mean", min_value=0.01, max_value=1.00, step=0.001
    )
    concavity_mean = st.number_input(
        "Concavity Mean", min_value=0.01, max_value=1.00, step=0.001
    )
    concave_points_mean = st.number_input(
        "Concave Points Mean", min_value=0.01, max_value=1.00, step=0.001
    )
    symmetry_mean = st.number_input(
        "Symmetry Mean", min_value=0.01, max_value=1.00, step=0.001
    )
    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean", min_value=0.01, max_value=1.00, step=0.001
    )

    radius_se = st.number_input("Radius SE", min_value=0.01, max_value=5.00, step=0.01)
    texture_se = st.number_input(
        "Texture SE", min_value=0.01, max_value=5.00, step=0.01
    )
    perimeter_se = st.number_input(
        "Perimeter SE", min_value=0.01, max_value=10.00, step=0.01
    )
    area_se = st.number_input("Area SE", min_value=0.01, max_value=100.00, step=0.01)
    smoothness_se = st.number_input(
        "Smoothness SE", min_value=0.001, max_value=1.00, step=0.0001
    )
    compactness_se = st.number_input(
        "Compactness SE", min_value=0.001, max_value=1.00, step=0.0001
    )
    concavity_se = st.number_input(
        "Concavity SE", min_value=0.001, max_value=1.00, step=0.0001
    )
    concave_points_se = st.number_input(
        "Concave Points SE", min_value=0.001, max_value=1.00, step=0.0001
    )
    symmetry_se = st.number_input(
        "Symmetry SE", min_value=0.001, max_value=1.00, step=0.0001
    )
    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE", min_value=0.001, max_value=1.00, step=0.0001
    )

    radius_worst = st.number_input(
        "Radius Worst", min_value=5.0, max_value=50.0, step=0.01
    )
    texture_worst = st.number_input(
        "Texture Worst", min_value=5.0, max_value=50.0, step=0.01
    )
    perimeter_worst = st.number_input(
        "Perimeter Worst", min_value=50.0, max_value=300.0, step=0.01
    )
    area_worst = st.number_input(
        "Area Worst", min_value=100.0, max_value=5000.0, step=1.0
    )
    smoothness_worst = st.number_input(
        "Smoothness Worst", min_value=0.01, max_value=1.00, step=0.001
    )
    compactness_worst = st.number_input(
        "Compactness Worst", min_value=0.01, max_value=1.00, step=0.001
    )
    concavity_worst = st.number_input(
        "Concavity Worst", min_value=0.01, max_value=1.00, step=0.001
    )
    concave_points_worst = st.number_input(
        "Concave Points Worst", min_value=0.01, max_value=1.00, step=0.001
    )
    symmetry_worst = st.number_input(
        "Symmetry Worst", min_value=0.01, max_value=1.00, step=0.001
    )
    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst", min_value=0.01, max_value=1.00, step=0.001
    )

    # --- Predict Breast Cancer ---
    if st.button("Predict Breast Cancer"):
        input_data = np.array(
            [
                [
                    radius_mean,
                    texture_mean,
                    perimeter_mean,
                    area_mean,
                    smoothness_mean,
                    compactness_mean,
                    concavity_mean,
                    concave_points_mean,
                    symmetry_mean,
                    fractal_dimension_mean,
                    radius_se,
                    texture_se,
                    perimeter_se,
                    area_se,
                    smoothness_se,
                    compactness_se,
                    concavity_se,
                    concave_points_se,
                    symmetry_se,
                    fractal_dimension_se,
                    radius_worst,
                    texture_worst,
                    perimeter_worst,
                    area_worst,
                    smoothness_worst,
                    compactness_worst,
                    concavity_worst,
                    concave_points_worst,
                    symmetry_worst,
                    fractal_dimension_worst,
                ]
            ]
        )
        input_data_scaled = breast_cancer_scaler.transform(input_data)
        prediction = breast_cancer_model.predict(input_data_scaled)[0]

        if prediction == 1:
            st.error("High risk of breast cancer!")
        else:
            st.success("No significant breast cancer detected.")

elif selected == "Diabetes Disease Prediction":
    # diabetes_disease_prediction.diabetes_disease_prediction()
    # Load model and scaler

    st.title("Diabetes Prediction Web App")
    st.markdown("### Enter details below to predict diabetes risk")

    # --- User Inputs ---
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=250)
    bp = st.number_input("Blood Pressure", min_value=30, max_value=200)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
    bmi = st.number_input("Body Mass Index", min_value=10.0, max_value=60.0, step=0.1)
    dp = st.number_input(
        "Diabetes Pedigree unction", min_value=0.0, max_value=3.0, step=0.01
    )
    age = st.number_input("Age", min_value=0, max_value=100)

    # --- Predict Diabetes ---
    if st.button("Predict Diabetes"):
        input_data = np.array(
            [[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dp, age]]
        )
        input_data_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_data_scaled)[0]

        if prediction == 1:
            st.error("High risk of diabetes!")
        else:
            st.success("No signiicant diabetes detected.")

elif selected == "Heart Disease Prediction":
    # heart_disease_prediction.heart_disease_prediction()
    st.title("Heart Disease Prediction Web App")
    st.markdown("### Enter the details below to predict heart disease risk")

    # --- User Inputs ---
    age = st.number_input("Age", min_value=0, max_value=100)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "Abnormal"])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", ["Type 1", "Type 2", "Type 3"]
    )
    ca = st.number_input(
        "Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4
    )
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # --- Categorical Mapping ---
    sex_map = {"Male": 1, "Female": 0}
    fbs_map = {"No": 0, "Yes": 1}
    exang_map = {"No": 0, "Yes": 1}
    thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
    restecg_map = {"Normal": 0, "Abnormal": 1}
    slope_map = {"Type 1": 0, "Type 2": 1, "Type 3": 2}
    cp_map = {"Type 1": 0, "Type 2": 1, "Type 3": 2, "Type 4": 3}

    # Convert categorical values to numbers
    sex = sex_map[sex]
    fbs = fbs_map[fbs]
    exang = exang_map[exang]
    thal = thal_map[thal]
    restecg = restecg_map[restecg]
    slope = slope_map[slope]
    cp = cp_map[cp]

    # --- Predict Heart Disease ---
    if st.button("Predict Heart Disease"):
        input_data = np.array(
            [
                [
                    age,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]
            ]
        )
        input_data_scaled = heart_scaler.transform(input_data)
        prediction = heart_model.predict(input_data_scaled)[0]

        if prediction == 1:
            st.error("High risk of heart disease!")
        else:
            st.success("No significant heart disease detected.")

elif selected == "Liver Disease Prediction":
    # liver_disease_prediction.liver_disease_prediction()

    st.title("Liver Disease Prediction Web App")
    st.markdown("### Enter details below to predict liver disease risk")

    # --- User Inputs ---
    age = st.number_input("Age", min_value=0, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input(
        "Total Bilirubin", min_value=0.0, max_value=20.0, step=0.1
    )
    direct_bilirubin = st.number_input(
        "Direct Bilirubin", min_value=0.0, max_value=10.0, step=0.1
    )
    total_proteins = st.number_input(
        "Total Proteins", min_value=0, max_value=1000, step=10
    )
    albumin = st.number_input("Albumin", min_value=0, max_value=100, step=10)
    ag_ratio = st.number_input(
        "Albumin/Globulin Ratio", min_value=0, max_value=100, step=10
    )
    sgpt = st.number_input(
        "SGPT (Alanine Aminotransferase)", min_value=0.1, max_value=10.0, step=0.1
    )
    sgot = st.number_input(
        "SGOT (Aspartate Aminotransferase)", min_value=0.1, max_value=10.0, step=0.1
    )
    alk_phos = st.number_input(
        "Alkaline Phosphotase", min_value=0.0, max_value=5.0, step=0.1
    )

    # --- Categorical Mapping ---
    gender_map = {"Male": 1, "Female": 0}
    gender = gender_map[gender]

    # --- Predict Liver Disease ---
    if st.button("Predict Liver Disease"):
        input_data = np.array(
            [
                [
                    age,
                    gender,
                    total_bilirubin,
                    direct_bilirubin,
                    total_proteins,
                    albumin,
                    ag_ratio,
                    sgpt,
                    sgot,
                    alk_phos,
                ]
            ]
        )
        input_data_scaled = liver_scaler.transform(input_data)
        prediction = liver_model.predict(input_data_scaled)[0]

        if prediction == 1:
            st.error("High risk of liver disease!")
        else:
            st.success("No significant liver disease detected.")

elif selected == "Parkinson's Disease Prediction":
    # parkinson_disease_prediction.parkinson_disease_prediction()

    st.title("Parkinson's Disease Prediction Web App")
    st.markdown("### Enter details below to predict Parkinson's disease risk")

    # --- User Inputs ---
    fo = st.number_input("MDVP - Fo(Hz)", min_value=50.0, max_value=300.0, step=0.01)
    fhi = st.number_input("MDVP - Fhi(Hz)", min_value=50.0, max_value=400.0, step=0.01)
    flo = st.number_input("MDVP - Flo(Hz)", min_value=50.0, max_value=300.0, step=0.01)
    jitter_percent = st.number_input(
        "MDVP - Jitter(%)", min_value=0.00, max_value=1.00, step=0.0001
    )
    jitter_abs = st.number_input(
        "MDVP - Jitter(Abs)", min_value=0.00000, max_value=0.01000, step=0.00001
    )
    rap = st.number_input("MDVP - RAP", min_value=0.000, max_value=0.100, step=0.0001)
    ppq = st.number_input("MDVP - PPQ", min_value=0.000, max_value=0.100, step=0.0001)
    ddp = st.number_input("Jitter - DDP", min_value=0.000, max_value=0.100, step=0.0001)
    shimmer = st.number_input(
        "MDVP - Shimmer", min_value=0.000, max_value=0.500, step=0.001
    )
    shimmer_db = st.number_input(
        "MDVP - Shimmer(dB)", min_value=0.00, max_value=10.00, step=0.01
    )
    apq3 = st.number_input(
        "Shimmer - APQ3", min_value=0.000, max_value=0.500, step=0.001
    )
    apq5 = st.number_input(
        "Shimmer - APQ5", min_value=0.000, max_value=0.500, step=0.001
    )
    mdvp_apq = st.number_input(
        "MDVP - APQ", min_value=0.000, max_value=0.500, step=0.001
    )
    dda = st.number_input("Shimmer - DDA", min_value=0.000, max_value=0.500, step=0.001)
    nhr = st.number_input("NHR", min_value=0.00, max_value=1.00, step=0.001)
    hnr = st.number_input("HNR", min_value=0.00, max_value=50.00, step=0.01)
    rpde = st.number_input("RPDE", min_value=0.00, max_value=1.00, step=0.001)
    dfa = st.number_input("DFA", min_value=0.00, max_value=1.00, step=0.001)
    spread1 = st.number_input("spread1", min_value=-10.00, max_value=0.00, step=0.01)
    spread2 = st.number_input("spread2", min_value=0.00, max_value=1.00, step=0.01)
    d2 = st.number_input("D2", min_value=0.00, max_value=5.00, step=0.01)
    ppe = st.number_input("PPE", min_value=0.00, max_value=1.00, step=0.01)

    # --- Predict Parkinson’s Disease ---
    if st.button("Predict Parkinson's Disease"):
        input_data = np.array(
            [
                [
                    fo,
                    fhi,
                    flo,
                    jitter_percent,
                    jitter_abs,
                    rap,
                    ppq,
                    ddp,
                    shimmer,
                    shimmer_db,
                    apq3,
                    apq5,
                    mdvp_apq,
                    dda,
                    nhr,
                    hnr,
                    rpde,
                    dfa,
                    spread1,
                    spread2,
                    d2,
                    ppe,
                ]
            ]
        )
        input_data_scaled = parkinson_scaler.transform(input_data)
        prediction = parkinson_model.predict(input_data_scaled)[0]

        if prediction == 1:
            st.error("High risk of Parkinson's disease!")
        else:
            st.success("No significant Parkinson's disease detected.")
