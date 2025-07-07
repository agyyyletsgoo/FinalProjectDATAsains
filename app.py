import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== Load model =====
model = joblib.load("random_forest_stroke_model.pkl")

# ===== Load custom CSS =====
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===== Judul =====
st.markdown("<h1>🧠 Prediksi Risiko Stroke</h1>", unsafe_allow_html=True)
st.markdown("Masukkan data pasien atau upload CSV untuk memprediksi risiko stroke.")

# ===== Mapping kategorikal =====
map_gender = {"Male": 1, "Female": 0, "Other": 2}
map_yesno = {"Tidak": 0, "Ya": 1, "No": 0, "Yes": 1}
map_work = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 1, "Never_worked": 4}
map_residence = {"Urban": 1, "Rural": 0}
map_smoke = {"formerly smoked": 1, "never smoked": 2, "smokes": 3, "Unknown": 0}

# ===== Input Manual =====
st.subheader("📋 Input Manual")

with st.form("input_form"):
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])
    age = st.slider("Usia", 0, 100, 30)
    hypertension = st.selectbox("Punya Hipertensi?", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Punya Penyakit Jantung?", ["Tidak", "Ya"])
    ever_married = st.selectbox("Sudah Menikah?", ["No", "Yes"])
    work_type = st.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Tempat Tinggal", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", value=90.0)
    bmi = st.number_input("BMI", value=22.0)
    smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("🔍 Prediksi")

    if submitted:
        input_data = pd.DataFrame({
            'id': [0],  # tambahkan kolom id dummy
            'gender': [map_gender[gender]],
            'age': [age],
            'hypertension': [map_yesno[hypertension]],
            'heart_disease': [map_yesno[heart_disease]],
            'ever_married': [map_yesno[ever_married]],
            'work_type': [map_work[work_type]],
            'Residence_type': [map_residence[Residence_type]],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'smoking_status': [map_smoke[smoking_status]]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Risiko Stroke Terdeteksi! Probabilitas: {probability:.2f}")
        else:
            st.success(f"✅ Tidak Terindikasi Stroke. Probabilitas: {probability:.2f}")
