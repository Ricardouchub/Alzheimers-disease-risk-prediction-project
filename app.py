import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        color: white;
        background-color: #1F618D;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1.2em;
    }
    .stButton>button:hover {
        background-color: #154360;
    }
    [data-testid="stSidebar"] {
        width: 400px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Models and Data ---
@st.cache_resource
def load_assets():
    model_pipeline = joblib.load('alzheimer_model_pipeline.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    X_train_data = joblib.load('X_train.pkl')
    return model_pipeline, kmeans_model, preprocessor, X_train_data

model, kmeans, preprocessor, X_train = load_assets()
X_train_processed = preprocessor.transform(X_train)

# --- Title ---
st.markdown("""
    <h1 style='text-align: center; color: #1F618D;'>🧠 Alzheimer's Risk & Phenotype Predictor</h1>
    <h4 style='text-align: center; color: grey;'>Predict and understand the potential risk using patient lifestyle, clinical and cognitive features</h4>
    <br>
""", unsafe_allow_html=True)

# --- Expander with Instructions ---
with st.expander("How to use", expanded=True):
    st.write("""
        This application uses a machine learning model to estimate Alzheimer's disease risk.

        **Instructions:**
        1. Fill out the form in the sidebar.
        2. Use "Unknown" if data is unavailable.
        3. Click "Predict Risk" to view results.

        **Results include:**
        - Risk Score
        - Waterfall Plot of contributing features
        - Phenotype categorization if high-risk
    """)

# --- Sidebar ---
st.sidebar.header("Patient Data Input")
st.sidebar.write("Use 'Unknown' if data is unavailable.")

# --- User Input Function ---
def user_input_features():
    def create_input_row(label, widget_type, options, default, key_suffix, col_widths=[3,2]):
        c1, c2 = st.sidebar.columns(col_widths)
        is_unknown = c2.checkbox("Unknown", key=f"{key_suffix}_unknown")
        if widget_type == 'slider':
            value = c1.slider(label, options[0], options[1], default, disabled=is_unknown, key=f"{key_suffix}_slider")
        elif widget_type == 'selectbox':
            value = c1.selectbox(label, options, index=options.index(default) if default in options else 0, disabled=is_unknown, key=f"{key_suffix}_selectbox")
        return np.nan if is_unknown else value

    st.sidebar.subheader("Demographics")
    Age = create_input_row('Age', 'slider', (60, 90), 75, 'age')
    Gender = create_input_row('Gender', 'selectbox', ('Male', 'Female'), 'Male', 'gender')
    Ethnicity = create_input_row('Ethnicity', 'selectbox', ('Caucasian', 'African American', 'Asian', 'Other'), 'Caucasian', 'ethnicity')
    EducationLevel = create_input_row('Education Level', 'selectbox', ('None', 'High School', "Bachelor's", 'Higher'), 'High School', 'education')

    st.sidebar.subheader("Lifestyle")
    BMI = create_input_row('BMI', 'slider', (15.0, 40.0), 25.0, 'bmi')
    Smoking = create_input_row('Smoking Status', 'selectbox', ('No', 'Yes'), 'No', 'smoking')
    AlcoholConsumption = create_input_row('Weekly Alcohol (units)', 'slider', (0, 20), 5, 'alcohol')
    PhysicalActivity = create_input_row('Weekly Physical Activity (hrs)', 'slider', (0.0, 10.0), 3.0, 'activity')
    DietQuality = create_input_row('Diet Quality (0-10)', 'slider', (0.0, 10.0), 5.0, 'diet')
    SleepQuality = create_input_row('Sleep Quality (4-10)', 'slider', (4.0, 10.0), 7.0, 'sleep')

    st.sidebar.subheader("Medical History & Measurements")
    FamilyHistoryAlzheimers = create_input_row("Family History", 'selectbox', ('No', 'Yes'), 'No', 'famhistory')
    CardiovascularDisease = create_input_row('Cardiovascular Disease', 'selectbox', ('No', 'Yes'), 'No', 'cardio')
    Diabetes = create_input_row('Diabetes', 'selectbox', ('No', 'Yes'), 'No', 'diabetes')
    Depression = create_input_row('Depression', 'selectbox', ('No', 'Yes'), 'No', 'depression')
    HeadInjury = create_input_row('Head Injury', 'selectbox', ('No', 'Yes'), 'No', 'headinjury')
    Hypertension = create_input_row('Hypertension', 'selectbox', ('No', 'Yes'), 'No', 'hypertension')
    SystolicBP = create_input_row('Systolic BP (mmHg)', 'slider', (90, 180), 120, 'sysbp')
    DiastolicBP = create_input_row('Diastolic BP (mmHg)', 'slider', (60, 120), 80, 'diabp')
    CholesterolTotal = create_input_row('Total Cholesterol', 'slider', (150, 300), 200, 'chtotal')
    CholesterolLDL = create_input_row('LDL Cholesterol', 'slider', (50, 200), 100, 'chldl')
    CholesterolHDL = create_input_row('HDL Cholesterol', 'slider', (20, 100), 50, 'chhdl')
    CholesterolTriglycerides = create_input_row('Triglycerides', 'slider', (50, 400), 150, 'chtrig')

    st.sidebar.subheader("Cognitive & Functional Assessments")
    MMSE = create_input_row('MMSE Score', 'slider', (0, 30), 25, 'mmse')
    FunctionalAssessment = create_input_row('Functional Assessment', 'slider', (0.0, 10.0), 8.0, 'funcassess')
    MemoryComplaints = create_input_row('Memory Complaints', 'selectbox', ('No', 'Yes'), 'No', 'memcomp')
    BehavioralProblems = create_input_row('Behavioral Problems', 'selectbox', ('No', 'Yes'), 'No', 'behavprob')
    ADL = create_input_row('ADL Score', 'slider', (0.0, 10.0), 9.0, 'adl')

    st.sidebar.subheader("Symptoms")
    Confusion = create_input_row('Confusion', 'selectbox', ('No', 'Yes'), 'No', 'confusion')
    Disorientation = create_input_row('Disorientation', 'selectbox', ('No', 'Yes'), 'No', 'disorientation')
    PersonalityChanges = create_input_row('Personality Changes', 'selectbox', ('No', 'Yes'), 'No', 'perschange')
    DifficultyCompletingTasks = create_input_row('Difficulty w/ Tasks', 'selectbox', ('No', 'Yes'), 'No', 'taskdiff')
    Forgetfulness = create_input_row('Forgetfulness', 'selectbox', ('No', 'Yes'), 'No', 'forgetful')

    data = {
        'Age': Age, 'Gender': np.nan if pd.isna(Gender) else (1 if Gender == 'Female' else 0),
        'Ethnicity': np.nan if pd.isna(Ethnicity) else ['Caucasian', 'African American', 'Asian', 'Other'].index(Ethnicity),
        'EducationLevel': np.nan if pd.isna(EducationLevel) else ['None', 'High School', "Bachelor's", 'Higher'].index(EducationLevel),
        'BMI': BMI, 'Smoking': np.nan if pd.isna(Smoking) else (1 if Smoking == 'Yes' else 0),
        'AlcoholConsumption': AlcoholConsumption, 'PhysicalActivity': PhysicalActivity,
        'DietQuality': DietQuality, 'SleepQuality': SleepQuality,
        'FamilyHistoryAlzheimers': np.nan if pd.isna(FamilyHistoryAlzheimers) else (1 if FamilyHistoryAlzheimers == 'Yes' else 0),
        'CardiovascularDisease': np.nan if pd.isna(CardiovascularDisease) else (1 if CardiovascularDisease == 'Yes' else 0),
        'Diabetes': np.nan if pd.isna(Diabetes) else (1 if Diabetes == 'Yes' else 0),
        'Depression': np.nan if pd.isna(Depression) else (1 if Depression == 'Yes' else 0),
        'HeadInjury': np.nan if pd.isna(HeadInjury) else (1 if HeadInjury == 'Yes' else 0),
        'Hypertension': np.nan if pd.isna(Hypertension) else (1 if Hypertension == 'Yes' else 0),
        'SystolicBP': SystolicBP, 'DiastolicBP': DiastolicBP,
        'CholesterolTotal': CholesterolTotal, 'CholesterolLDL': CholesterolLDL,
        'CholesterolHDL': CholesterolHDL, 'CholesterolTriglycerides': CholesterolTriglycerides,
        'MMSE': MMSE, 'FunctionalAssessment': FunctionalAssessment,
        'MemoryComplaints': np.nan if pd.isna(MemoryComplaints) else (1 if MemoryComplaints == 'Yes' else 0),
        'BehavioralProblems': np.nan if pd.isna(BehavioralProblems) else (1 if BehavioralProblems == 'Yes' else 0),
        'ADL': ADL, 'Confusion': np.nan if pd.isna(Confusion) else (1 if Confusion == 'Yes' else 0),
        'Disorientation': np.nan if pd.isna(Disorientation) else (1 if Disorientation == 'Yes' else 0),
        'PersonalityChanges': np.nan if pd.isna(PersonalityChanges) else (1 if PersonalityChanges == 'Yes' else 0),
        'DifficultyCompletingTasks': np.nan if pd.isna(DifficultyCompletingTasks) else (1 if DifficultyCompletingTasks == 'Yes' else 0),
        'Forgetfulness': np.nan if pd.isna(Forgetfulness) else (1 if Forgetfulness == 'Yes' else 0),
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Prediction ---
st.markdown("## 🧪 Prediction Results")
col1, col2 = st.columns([1, 2])

if st.sidebar.button('🔍 Predict Risk'):
    prediction_proba = model.predict_proba(input_df)
    risk_score = prediction_proba[0][1]

    with col1:
        st.metric(label="Alzheimer's Risk Score", value=f"{risk_score:.0%}")
        if risk_score > 0.6:
            st.error("🔴 High Risk")
        elif risk_score > 0.4:
            st.warning("🟡 Moderate Risk")
        else:
            st.success("🟢 Low Risk")

    with col2:
        st.subheader("Prediction Explanation")
        explainer = shap.TreeExplainer(model.named_steps['classifier'], X_train_processed)
        input_processed = model.named_steps['preprocessor'].transform(input_df)
        shap_values = explainer.shap_values(input_processed)

        fig, ax = plt.subplots(figsize=(9, 4))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value, 
                                             data=input_processed[0],
                                             feature_names=preprocessor.get_feature_names_out()), 
                            show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("This plot shows how features push the risk up or down.")

    if risk_score > 0.5:
        st.markdown("### 🧬 Patient Phenotype Profile")
        patient_cluster = kmeans.predict(input_processed)[0]
        phenotype_descriptions = {
            0: "**🍔 Phenotype A: Metabolic-Dominant**",
            1: "**🧠 Phenotype B: Cognitive-Impairment Dominant**",
            2: "**🚬 Phenotype C: Lifestyle-Risk**"
        }
        st.markdown(phenotype_descriptions.get(patient_cluster, "No specific phenotype identified."))
else:
    st.info("Use the sidebar to input patient data and click **Predict Risk**.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p style='color: grey;'>Developed by <b>Ricardo Urdaneta</b></p>
    <a href="https://github.com/Ricardouchub" target="_blank" style="text-decoration:none;">
        <button style='margin-right: 10px;' class="footer-btn">GitHub</button>
    </a>
    <a href="https://www.linkedin.com/in/ricardourdanetacastro" target="_blank" style="text-decoration:none;">
        <button class="footer-btn">LinkedIn</button>
    </a>
</div>
<style>
.footer-btn {
    background-color: #1F618D;
    border: none;
    color: white;
    padding: 8px 16px;
    text-align: center;
    border-radius: 8px;
    font-size: 14px;
    margin-top: 10px;
    cursor: pointer;
}
.footer-btn:hover {
    background-color: #154360;
}
</style>
""", unsafe_allow_html=True)
