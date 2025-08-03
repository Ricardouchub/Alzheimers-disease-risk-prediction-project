# Alzheimer's Disease Risk Predictor

<p align="left">
  <img src="https://img.shields.io/badge/Project_Status-Completed-green?style=for-the-badge" alt="Project Status: Completed"/>
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn" alt="scikit-learn"/>
</p>

This project demonstrates an end-to-end data science workflow to develop and deploy an interactive tool for assessing the risk of Alzheimer's disease. The application uses a machine learning model to provide a risk score, identifies distinct patient profiles through clustering, and explains its predictions using SHAP.

---

**[Check Notebook](https://github.com/Ricardouchub/Alzheimers-disease-risk-prediction-project/blob/main/Notebook.ipynb)**

### **[Live web app](https://alzheimers-disease-risk-prediction-project.streamlit.app/)**

<img width="747" height="364" alt="image" src="https://github.com/user-attachments/assets/79991e09-9575-4504-9d90-318d72e3c52a" />

**Dataset: [Kaggle](https://www.kaggle.com/dsv/8668279)**

---

### Table of Contents
1. [Project Description](#project-description)
2. [Key Features](#key-features)
3. [Phases](#phases)
4. [Tools](#tools)
5. [How to Run Locally](#how-to-run-locally)
6. [Project Structure](#project-structure)

---

### 1. Project Description

This project aims to develop a comprehensive tool that moves beyond a simple diagnostic prediction. Instead of just answering "Does this patient have Alzheimer's?", this interactive platform provides a nuanced **Risk Score**, identifies underlying **Patient Profiles (Phenotypes)**, and offers transparent, **Explainable Insights** into its predictions, mimicking a valuable tool for a healthcare professional.

---

### 2. Key Features

- **Predictive Risk Scoring:** Utilizes an XGBoost model to generate a precise probability score (0-100%) of Alzheimer's risk.
- **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) to create a visual breakdown of which factors contributed to each prediction.
- **Unsupervised Patient Clustering:** Employs K-Means clustering to identify and assign high-risk patients to distinct clinical profiles (e.g., Metabolic-Dominant, Cognitive-Dominant).
- **Interactive Web Interface:** A user-friendly dashboard built with Streamlit that allows for real-time data input and analysis.
- **Robust Data Handling:** The application is designed to handle missing data points, allowing users to make predictions even with incomplete information.

---

### 3. Phases

The project was structured into the following phases:

1.  **Exploratory Data Analysis (EDA) and Preprocessing:**
    The initial dataset was loaded, cleaned, and processed. This involved handling missing values with `SimpleImputer`, encoding categorical data, and scaling numerical features. Extensive visualizations were created to understand feature distributions and identify key correlations with the Alzheimer's diagnosis.

2.  **Predictive Modeling and "Risk Score" Development:**
    Several machine learning models were trained and evaluated, with **XGBoost** selected for its high performance. Instead of a binary prediction, the model was designed to output a nuanced **Risk Score** from prediction probabilities.

3.  **Patient Profiling (Clustering):**
    **K-Means** clustering was applied to the subset of diagnosed patients. This phase aimed to discover underlying data structures and identify distinct patient **phenotypes** based on their clinical and lifestyle profiles.

4.  **Interactive Dashboard:**
    A user-friendly web application was built using **Streamlit**. This dashboard integrates the predictive model and allows users to input patient data to receive a real-time risk assessment. Crucially, it incorporates **SHAP** to generate a waterfall plot that explains every prediction, making the model's decisions transparent.

5.  **Deployment:**
    The final application, including the trained models and all necessary assets, was packaged and deployed to **Streamlit Community Cloud**, making it a publicly accessible tool.

---

### 4. Tools

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Web Framework:** Streamlit
- **Data Visualization:** Matplotlib, Seaborn

---

### 5. How to Run Locally

To run this application on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ricardouchub/Alzheimers-disease-risk-prediction-project.git
    cd Alzheimers-disease-risk-prediction-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

---

### 6. Project Structure

- **`Notebook.ipynb`**: Main notebook with EDA, visualizations and models training and evaluation.
- **`models/`**: Contains all the serialized, pre-trained models and objects.
- **`app.py`**: The main script for the Streamlit web application.
- **`requirements.txt`**: A list of all Python libraries required to run the project.

---

### Author:
**Ricardo Urdaneta** 

[**LinkedIn**](https://www.linkedin.com/in/ricardourdanetacastro)



