import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("SVM.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("columns.pkl")

st.title("🔬 Breast Cancer Diagnosis Predictor by Satyam")
st.write("Enter the patient's features to predict if the tumor is *Benign (B)* or *Malignant (M)*.")

# Streamlit form for input
with st.form("input_form"):
    inputs = {}
    for col in feature_columns:
        inputs[col] = st.number_input(col, min_value=0.0, format="%.5f")
    submit = st.form_submit_button("🔍 Predict")

if submit:
    # Convert inputs into DataFrame with correct column order
    input_df = pd.DataFrame([[inputs[col] for col in feature_columns]], columns=feature_columns)

    # Scale inputs
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠ Prediction: *Malignant* (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Prediction: *Benign* (Probability: {1-proba:.2f})")