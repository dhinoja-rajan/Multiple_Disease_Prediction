import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import joblib

diabetes_model = joblib.load("./saved_models/DIABETES_Disease_Prediction_Model.pkl")
diabetes_scaler = joblib.load("./saved_models/diabetes_scaler.pkl")

heart_model = joblib.load("./saved_models/HEART_Disease_Prediction_Model.pkl")
heart_scaler = joblib.load("./saved_models/heart_disease_scaler.pkl")

liver_model = joblib.load("./saved_models/LIVER_Disease_Prediction_Model.pkl")
liver_scaler = joblib.load("./saved_models/liver_disease_scaler.pkl")


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
            "Diabetes Disease Prediction",
            "Heart Disease Prediction",
            "Liver Disease Prediction",
        ],
        menu_icon="hospital-fill",
        icons=[
            "house-fill",
            "activity",
            "heart-fill",
            "droplet-fill",
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
        '<p class="big-title">🔬 AI-Powered Multiple Disease Prediction</p>',
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
        # st.markdown("---")
    with col2:
        st.metric("🦠 Diabetes", "8.5M cases/year", "Worldwide epidemic")
    with col3:
        st.metric("🩸 Liver Disease", "1.2M cases/year", "Increasing Cases worldwide")

    st.markdown("---")

    # Title and subtitle
    st.markdown('<p class="quiz-title">🧠 AI Health Quiz</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="quiz-subtitle">Test your knowledge about health and diseases!</p>',
        unsafe_allow_html=True,
    )

    # Dictionary of questions and answers for multiple disease prediction project
    questions = {
        "What is the normal range of blood pressure": {
            "options": ["120/80 mmHg", "150/100 mmHg", "90/60 mmHg"],
            "answer": "120/80 mmHg",
        },
        "Which type of diabetes occurs due to insulin resistance": {
            "options": ["Type 1", "Type 2", "Gestational"],
            "answer": "Type 2",
        },
        # Heart Disease Questions
        "What is the most common cause of heart disease?": {
            "options": ["High blood pressure", "Low cholesterol", "Dehydration"],
            "answer": "High blood pressure",
        },
        "Which lifestyle change can help reduce the risk of heart disease?": {
            "options": [
                "Smoking",
                "Exercising regularly",
                "Excessive alcohol consumption",
            ],
            "answer": "Exercising regularly",
        },
        "Which test is commonly used to diagnose heart disease?": {
            "options": ["Electrocardiogram (ECG)", "Ultrasound", "MRI"],
            "answer": "Electrocardiogram (ECG)",
        },
        # Liver Disease Questions
        "What is the most common cause of liver disease?": {
            "options": [
                "Excessive alcohol consumption",
                "Obesity",
                "Physical inactivity",
            ],
            "answer": "Excessive alcohol consumption",
        },
        "Which of the following is a symptom of liver disease?": {
            "options": [
                "Yellowing of the skin and eyes",
                "Shortness of breath",
                "Headache",
            ],
            "answer": "Yellowing of the skin and eyes",
        },
        "What test is commonly used to check liver function?": {
            "options": ["Liver function tests (LFT)", "CT scan", "X-ray"],
            "answer": "Liver function tests (LFT)",
        },
        # Diabetes Questions
        "What is the primary symptom of diabetes?": {
            "options": ["Excessive thirst and hunger", "Dry skin", "Headache"],
            "answer": "Excessive thirst and hunger",
        },
        "Which of the following is a risk factor for developing Type 2 diabetes?": {
            "options": ["High blood pressure", "Low cholesterol", "Frequent urination"],
            "answer": "High blood pressure",
        },
        "What type of diabetes requires daily insulin injections?": {
            "options": ["Type 1", "Type 2", "Gestational"],
            "answer": "Type 1",
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
    st.markdown("### 🏥 Select a disease from the sidebar and Start Predicting!...")
    st.markdown("---")

elif selected == "Diabetes Disease Prediction":
    # diabetes_disease_prediction.diabetes_disease_prediction()
    # Load model and scaler

    st.title("Diabetes Prediction Web App")
    st.markdown("### Enter details below to predict diabetes risk")

    # --- User Inputs ---
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=17)
    glucose = st.number_input("Glucose Level(mg/dl)", min_value=50, max_value=199)
    bp = st.number_input("Blood Pressure(mm Hg)", min_value=30, max_value=122)
    skin_thickness = st.number_input("Skin Thickness(mm)", min_value=0, max_value=99)
    insulin = st.number_input("Insulin Level(mu U/ml)", min_value=0, max_value=846)
    bmi = st.number_input("Body Mass Index", min_value=0.0, max_value=67.1, step=0.1)
    dp = st.number_input(
        "Diabetes Pedigree unction", min_value=0.078, max_value=2.42, step=0.01
    )
    age = st.number_input("Age", min_value=21, max_value=100)

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
    age = st.number_input("Age", min_value=29, max_value=100)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Type 1", "Type 2", "Type 3", "Type 4"])
    trestbps = st.number_input(
        "Resting Blood Pressure(mm Hg)", min_value=94, max_value=200
    )
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "Abnormal"])
    thalach = st.number_input(
        "Max Heart Rate Achieved(bpm)", min_value=71, max_value=202
    )
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, step=0.1
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", ["Type 1", "Type 2", "Type 3"]
    )
    ca = st.number_input(
        "Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, step=1
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
    age = st.number_input("Age", min_value=20, max_value=100)
    gender = st.selectbox("Gender", ["Male", "Female"])
    total_bilirubin = st.number_input(
        "Total Bilirubin(mg/dL)", min_value=0.1, max_value=20.0, step=0.1
    )
    direct_bilirubin = st.number_input(
        "Direct Bilirubin(mg/dL)", min_value=0.1, max_value=10.0, step=0.1
    )
    total_proteins = st.number_input(
        "Total Proteins(g/dL)", min_value=100, max_value=1000, step=10
    )
    albumin = st.number_input("Albumin(g/dL)", min_value=10, max_value=100, step=10)
    ag_ratio = st.number_input(
        "Albumin/Globulin Ratio", min_value=10, max_value=100, step=10
    )
    sgpt = st.number_input(
        "SGPT (Alanine Aminotransferase)(U/L)", min_value=0.1, max_value=10.0, step=0.1
    )
    sgot = st.number_input(
        "SGOT (Aspartate Aminotransferase)(U/L)",
        min_value=0.1,
        max_value=10.0,
        step=0.1,
    )
    alk_phos = st.number_input(
        "Alkaline Phosphotase(IU/L)", min_value=0.1, max_value=5.0, step=0.1
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
