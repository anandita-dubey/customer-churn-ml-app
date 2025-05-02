# churn_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# Load feature columns from training
feature_columns = pickle.load(open("features.pkl", "rb"))

# Title
st.title("🔍 Customer Churn Prediction App")

st.markdown("Enter customer details below to predict whether they are likely to churn.")

# --- Input Widgets ---
gender = st.selectbox("Gender", ['Female', 'Male'])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ['Yes', 'No'])
Dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
MultipleLines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
PaymentMethod = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check',
    'Bank transfer (automatic)', 'Credit card (automatic)'
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)

# --- Convert to DataFrame ---
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_df = pd.DataFrame([input_dict])

# --- One-hot Encoding (same as training) ---
input_processed = pd.get_dummies(input_df)

# Align with training features
input_processed = input_processed.reindex(columns=feature_columns, fill_value=0)

# --- Prediction ---
if st.button("Predict Churn"):
    prediction = model.predict(input_processed)[0]
    result = "⚠️ Customer will likely churn." if prediction else "✅ Customer is likely to stay."
    st.subheader(result)
