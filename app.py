import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import joblib

# --- 1. Page Configuration ---
st.set_page_config(page_title="Alzheimer Risk Predictor", layout="wide")

# --- Custom CSS Styling ---
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
    .css-1d391kg .e1fqkh3o10 {
        background-color: #1F618D !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Header ---
st.markdown("""
    <h1 style='text-align: center; color: #1F618D;'>Alzheimer's Risk & Phenotype Predictor</h1>
    <h4 style='text-align: center; color: grey;'>Predict and understand the potential risk using patient lifestyle, clinical and cognitive features</h4>
    <br>
""", unsafe_allow_html=True)

# --- 3. Load Model and Data ---
model = joblib.load("model.pkl")
kmeans = joblib.load("kmeans_model.pkl")
preprocessor = model.named_steps['preprocessor']
X_train_processed = joblib.load("X_train_processed.pkl")

# --- Sidebar Inputs ---
st.sidebar.header("Patient Information")

def user_input_features():
    age = st.sidebar.slider('Age', 50, 90, 65)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    apoe = st.sidebar.selectbox('APOE Genotype', ['E3/E3', 'E3/E4', 'E4/E4'])
    education = st.sidebar.slider('Years of Education', 0, 20, 12)
    bmi = st.sidebar.slider('BMI', 15.0, 40.0, 25.0)
    diet = st.sidebar.slider('Diet Quality (1=Poor, 5=Excellent)', 1, 5, 3)
    sleep = st.sidebar.slider('Sleep Quality (1=Poor, 5=Excellent)', 1, 5, 3)
    exercise = st.sidebar.slider('Exercise Hours/Week', 0, 15, 3)
    memory = st.sidebar.slider('Memory Score (0-100)', 0, 100, 70)

    data = {
        'Age': age,
        'Gender': gender,
        'APOE': apoe,
        'Education': education,
        'BMI': bmi,
        'Diet': diet,
        'Sleep': sleep,
        'Exercise': exercise,
        'Memory': memory
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. Main Panel: Prediction and Explanation ---
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

    # --- Patient phenotype ---
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
