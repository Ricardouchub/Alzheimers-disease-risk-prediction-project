# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Load the trained models and preprocessor
@st.cache_resource
def load_assets():
    """Loads the trained models and preprocessor."""
    # Add 'models/' to the path of each file
    model_pipeline = joblib.load('models/alzheimer_model_pipeline.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    X_train_data = joblib.load('models/X_train.pkl')
    return model_pipeline, kmeans_model, preprocessor, X_train_data

model, kmeans, preprocessor, X_train = load_assets()

# Pre-process the training data once for the SHAP explainer
X_train_processed = preprocessor.transform(X_train)

# App Title and Introduction
st.title("Alzheimer's Disease Risk & Phenotype Predictor")

# Project Description
with st.expander("How to use", expanded=True):
    st.write("""
        This application leverages a machine learning model to provide a comprehensive risk assessment for Alzheimer's disease based on a patient's clinical, demographic, and lifestyle data. Beyond a simple prediction, it also offers two key insights:
        
        1.  Factors that contributes to a patient's risk score.
        2.  Patient Phenotyping for high-risk individuals, helping to understand the underlying nature of their risk.

        **Instructions:**
        1.  Use the sidebar on the left to input the patient's data.
        2.  If a piece of information is not available, simply check the **"Unknown"** box next to it. The model is trained to handle missing data.
        3.  Click the **"Predict"** button at the bottom of the sidebar to see the results.

        **Results Explained:**
        - **Alzheimer's Risk Score:** A percentage indicating the model's estimated probability of an Alzheimer's diagnosis.
        - **Prediction Explanation:** A waterfall plot showing the factors that increase risk (in red) and decrease risk (in blue).
        - **Patient Phenotype Profile:** For patients with a risk score above 50%, a potential clinical profile is suggested based on patterns found in the data: 
             
             **Phenotype A:** Metabolic-Dominant, **Phenotype B:** Cognitive-Impairment Dominant or **Phenotype C:** Lifestyle-Risk.
    """)

# Sidebar Header
st.sidebar.header("Patient Data Input")
st.sidebar.write("For any unavailable data, check the 'Unknown' box.")

#  User Input Features 
def user_input_features():
    """Creates sidebar widgets for user input and returns a DataFrame."""
    
    # Helper function to create input rows 
    def create_input_row(label, widget_type, options, default, key_suffix, col_widths=[3,2]):
        c1, c2 = st.sidebar.columns(col_widths)
        is_unknown = c2.checkbox("Unknown", key=f"{key_suffix}_unknown")
        
        if widget_type == 'slider':
            value = c1.slider(label, options[0], options[1], default, disabled=is_unknown, key=f"{key_suffix}_slider")
        elif widget_type == 'selectbox':
            value = c1.selectbox(label, options, index=options.index(default) if default in options else 0, disabled=is_unknown, key=f"{key_suffix}_selectbox")
        
        return np.nan if is_unknown else value

    # Create all inputs using the helper function 
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

    # Create the dictionary for the DataFrame
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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel: Prediction and Explanation
st.header("Prediction Results")

if st.sidebar.button('Predict'):
    # Get prediction probability
    prediction_proba = model.predict_proba(input_df)
    risk_score = prediction_proba[0][1]

    # Display Risk Score 
    st.subheader(f"Alzheimer's Risk Score: {risk_score:.0%}")
    if risk_score > 0.6:
        st.error("High Risk")
    elif risk_score > 0.4:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")

    # Display SHAP Explanation 
    st.subheader("Prediction Explanation")
    
    explainer = shap.TreeExplainer(model.named_steps['classifier'], X_train_processed)
    input_processed = model.named_steps['preprocessor'].transform(input_df)
    shap_values = explainer.shap_values(input_processed)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                          base_values=explainer.expected_value, 
                                          data=input_processed[0],
                                          feature_names=preprocessor.get_feature_names_out()), 
                        show=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.write("The waterfall plot shows which factors push the risk score higher (red) or lower (blue).")

    # Display Phenotype Assignment
    if risk_score > 0.5:
        st.subheader("Patient Phenotype Profile")
        patient_cluster = kmeans.predict(input_processed)[0]
        phenotype_descriptions = {
            0: "**Phenotype A: Metabolic-Dominant** 🍔",
            1: "**Phenotype B: Cognitive-Impairment Dominant** 🧠",
            2: "**Phenotype C: Lifestyle-Risk** 🚬"
        }
        st.markdown(phenotype_descriptions.get(patient_cluster, "No specific phenotype identified."))
else:
    st.info("Use the sidebar to input patient data and click 'Predict' to see the results.")

# --- CSS to Style the App ---
# CSS to Widen the Sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 400px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- FOOTER ---
st.markdown("---")

# HTML for the footer
footer_html = """
<div style="text-align: center;">
    <p>Author: Ricardo Urdaneta</p>
    <a href="https://github.com/Ricardouchub" target="_blank">
        <button class="footer-btn">Github</button>
    </a>
    <a href="https://www.linkedin.com/in/ricardourdanetacastro" target="_blank">
        <button class="footer-btn">Linkedin</button>
    </a>
</div>
"""

# CSS for the footer buttons ONLY
footer_css = """
<style>
.footer-btn {
    background-color: transparent;
    color: #1F618D;
    padding: 8px 20px;
    border-radius: 8px;
    border: 2px solid #1F618D;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    transition-duration: 0.4s;
}
.footer-btn:hover {
    background-color: #1F618D;
    color: white;
}
</style>
"""

# Render the footer
st.markdown(footer_html, unsafe_allow_html=True)
st.markdown(footer_css, unsafe_allow_html=True)