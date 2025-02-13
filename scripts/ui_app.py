"""
ui_app.py

Interfaccia minimale con Streamlit per inserire i dati di un utente
e prevedere il rischio di ictus (stroke).
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Importiamo il path del modello e degli encoders dal config
from scripts.config import UI_MODEL_PATH, UI_ENCODER_PATH


def load_model(model_path: str):
    """Carica il modello Random Forest addestrato."""
    return joblib.load(model_path)

def load_encoders(encoder_path: str):
    """Carica i LabelEncoder usati per la trasformazione delle feature categoriali."""
    return joblib.load(encoder_path)

def main():
    st.title("Stroke Prediction App")
    st.write("Compila i campi sottostanti per ottenere una previsione sul rischio di ictus.")

    # Carica modello e LabelEncoders
    model = load_model(UI_MODEL_PATH)
    encoders = load_encoders(UI_ENCODER_PATH)

    # Input utente
    age = st.slider("Age:", min_value=0, max_value=100, value=30)
    hypertension = st.selectbox("Hypertension:", [0, 1])  # 0 = No, 1 = Yes
    heart_disease = st.selectbox("Heart Disease:", [0, 1])  # 0 = No, 1 = Yes
    ever_married = st.selectbox("Ever Married:", [0, 1])
    work_type = st.selectbox("Work Type:", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    avg_glucose_level = st.number_input("Average Glucose Level:", min_value=0.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI:", min_value=0.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status:", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Bottone per predizione
    if st.button("Predict Stroke Risk"):
        # Creazione del DataFrame con i nomi delle feature
        feature_names = [
            "age", "hypertension", "heart_disease", "ever_married",
            "work_type", "avg_glucose_level", "bmi", "smoking_status"
        ]

        input_data = pd.DataFrame([[
            age,
            hypertension,
            heart_disease,
            ever_married,
            encoders['work_type'].transform([work_type])[0],
            avg_glucose_level,
            bmi,
            encoders['smoking_status'].transform([smoking_status])[0]
        ]], columns=feature_names)

        # Predizione
        prediction = model.predict(input_data)[0]  # 0 = no stroke, 1 = stroke
        prediction_proba = model.predict_proba(input_data)[0][1] * 100  # probabilità di stroke

        # Mostra il risultato
        if prediction == 1:
            st.error(f"**High Risk**: La probabilità stimata di ictus è ~{prediction_proba:.2f}%")
        else:
            st.success(f"**Low Risk**: La probabilità stimata di ictus è ~{prediction_proba:.2f}%")

if __name__ == "__main__":
    main()