import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

SCALER_PATH = 'scaler.save'
MODEL_PATH = 'customer_churn_model.keras'
DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

@st.cache_resource
def get_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error(f"Model file '{MODEL_PATH}' not found. Train model and save as .h5!")
        return None

@st.cache_data
def get_data():
    return pd.read_csv(DATA_PATH)

def preprocess(df):
    df = df.copy()
    for col in ['PhoneService', 'MultipleLines']:
        if col in df.columns:
            df[col] = df[col].replace({'No phone service': 'No'})
    for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No'})
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=[col for col in cat_cols if col in df.columns])
    expected_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    scaler = get_scaler()
    scaled_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[scaled_cols] = scaler.transform(df[scaled_cols])
    return df

def risk_statement(prob):
    if prob < 0.2:
        return "This customer is very likely to stay."
    elif prob < 0.5:
        return "This customer is likely to stay."
    elif prob < 0.7:
        return "This customer has a moderate risk of leaving."
    else:
        return "This customer is at high risk of leaving the service."

def predict_customer(input_df, model):
    X_processed = preprocess(input_df)
    pred_prob = model.predict(X_processed)[0][0]
    pred_label = 'Churn' if pred_prob >= 0.5 else 'No Churn'
    return pred_label, pred_prob

def batch_predict(input_df, model):
    X_processed = preprocess(input_df)
    probs = model.predict(X_processed).flatten()
    labels = np.where(probs >= 0.5, 'Churn', 'No Churn')
    result_df = input_df.copy()
    result_df['Churn_Prediction'] = labels
    result_df['Probability'] = probs
    return result_df

st.title('Customer Churn Prediction App')
st.markdown("Predict telecom customer churn and display a clear risk statement for every probability.")

sidebar_mode = st.sidebar.radio('Select Mode:', ['Single Customer Prediction', 'Batch Prediction'])

if sidebar_mode == 'Single Customer Prediction':
    st.header('Customer Details')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    SeniorCitizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    Partner = st.selectbox('Partner', ['No', 'Yes'])
    Dependents = st.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12)
    PhoneService = st.selectbox('Phone Service', ['No', 'Yes'])
    MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No internet service'])
    InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.selectbox('Paperless Billing', ['No', 'Yes'])
    PaymentMethod = st.selectbox('Payment Method', [
        'Bank transfer (automatic)', 'Credit card (automatic)',
        'Electronic check', 'Mailed check'
    ])
    MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, value=70.0)
    TotalCharges = st.number_input('Total Charges', min_value=0.0, value=1000.0)

    if st.button('Predict Churn'):
        user_input = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges],
        })
        model = get_model()
        if model:
            label, prob = predict_customer(user_input, model)
            st.session_state['prediction_label'] = label
            st.session_state['prediction_prob'] = prob

    if 'prediction_label' in st.session_state and 'prediction_prob' in st.session_state:
        label = st.session_state['prediction_label']
        prob = st.session_state['prediction_prob']
        st.markdown(f"<span style='color:{'red' if label == 'Churn' else 'green'}; font-size:24px'>{label}</span>", unsafe_allow_html=True)
        st.success(f"Probability: {prob:.2f}")
        st.info(risk_statement(prob))

elif sidebar_mode == 'Batch Prediction':
    st.header('Bulk Prediction from CSV')
    uploaded_file = st.file_uploader('Upload CSV file (same columns as original dataset)', type=['csv'])
    if uploaded_file:
        input_batch = pd.read_csv(uploaded_file)
        model = get_model()
        if model:
            results = batch_predict(input_batch, model)
            st.dataframe(results)
            st.download_button('Download Results as CSV', results.to_csv(index=False), file_name='churn_predictions.csv')
