import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# 🚀 Page Configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# 🎓 Title and Description with custom color
st.markdown("<h1 style='text-align: center; color: #2D5BE3;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color:#444;'>Enter student details below and get a prediction for their performance.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-top: 2px solid #2D5BE3;'>", unsafe_allow_html=True)

# 📥 Sidebar Inputs
st.sidebar.header("📋 Student Information")
age = st.sidebar.number_input("Age", min_value=5, max_value=25, value=16)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
ethnicity = st.sidebar.selectbox("Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
parent_edu = st.sidebar.selectbox("Parental Education", ["None", "High School", "Diploma", "Bachelor", "Master"])
study_time = st.sidebar.slider("Weekly Study Time (hours)", 0.0, 40.0, 10.0)
absences = st.sidebar.number_input("Absences", min_value=0, max_value=100, value=3)
tutoring = st.sidebar.selectbox("Tutoring Received", ["No", "Yes"])
support = st.sidebar.selectbox("Parental Support", ["No", "Yes"])
extra = st.sidebar.selectbox("Extracurricular Activities", ["No", "Yes"])
sports = st.sidebar.selectbox("Plays Sports", ["No", "Yes"])
music = st.sidebar.selectbox("Plays Music", ["No", "Yes"])
volunteer = st.sidebar.selectbox("Volunteering", ["No", "Yes"])

# 🌐 Convert inputs to numerical format (modify as per your encoding)
input_data = pd.DataFrame([[
    age,
    0 if gender == "Male" else 1,
    ["Group A", "Group B", "Group C", "Group D", "Group E"].index(ethnicity),
    ["None", "High School", "Diploma", "Bachelor", "Master"].index(parent_edu),
    study_time,
    absences,
    0 if tutoring == "No" else 1,
    0 if support == "No" else 1,
    0 if extra == "No" else 1,
    0 if sports == "No" else 1,
    0 if music == "No" else 1,
    0 if volunteer == "No" else 1
]], columns=[
    'Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
    'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular',
    'Sports', 'Music', 'Volunteering'
])

# 🎯 Scaled Prediction
input_scaled = scaler.transform(input_data)

st.markdown("### 🔮 Prediction Result")
if st.button("📊 Predict Performance"):
    prediction = model.predict(input_scaled)[0]
    label_map = {0: "Slow Bloomer", 1: "Good Going", 2: "Fast Bloomer"}
    st.success(f"🎓 Predicted Output: **{label_map.get(prediction, prediction)}**")
    
    # Optionally add helpful note
    st.markdown(f"<div style='background-color:#EDF6FF; padding:20px; border-radius:10px; color:#0B5394; text-align:center;'>🎯 <strong>Predicted Output:</strong> {prediction}</div>", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid #BBB;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 13px; color:#888;'>Made with ❤️ by Srisubitsa | Powered by Machine Learning + Streamlit</p>", unsafe_allow_html=True)
