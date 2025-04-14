import pandas as pd
import pickle
import streamlit as st

# Streamlit app title and description
st.set_page_config(page_title="Loan Eligibility Predictor")
st.title("Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# Load the pre-trained Random Forest model
try:
    with open("RFmodel.pkl", "rb") as rf_pickle:
        rf_model = pickle.load(rf_pickle)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# User input form
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")

    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Marital Status", ["Yes", "No"])
    Dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

    ApplicantIncome = st.number_input("Applicant Monthly Income", min_value=0, step=1000)
    CoapplicantIncome = st.number_input("Coapplicant Monthly Income", min_value=0, step=1000)
    LoanAmount = st.number_input("Loan Amount", min_value=0, step=1000)
    Loan_Amount_Term = int(st.selectbox("Loan Amount Term (Months)", ["360", "180", "240", "120", "60"]))
    Credit_History = int(st.selectbox("Credit History", ["1", "0"]))
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Loan Eligibility")

# Encode categorical variables
if submitted:
    Gender_Male = 1 if Gender == "Male" else 0
    Married_Yes = 1 if Married == "Yes" else 0

    Dependents_1 = 1 if Dependents == "1" else 0
    Dependents_2 = 1 if Dependents == "2" else 0
    Dependents_3 = 1 if Dependents == "3+" else 0

    Education_Not_Graduate = 1 if Education == "Not Graduate" else 0
    Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0

    Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
    Property_Area_Urban = 1 if Property_Area == "Urban" else 0

    input_data = [[
        ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
        Credit_History, Gender_Male, Married_Yes,
        Dependents_1, Dependents_2, Dependents_3,
        Education_Not_Graduate, Self_Employed_Yes,
        Property_Area_Semiurban, Property_Area_Urban
    ]]

    try:
        new_prediction = rf_model.predict(input_data)
        st.subheader("Prediction Result:")
        if new_prediction[0] == 1:
            st.success("✅ You are eligible for a loan.")
        else:
            st.error("❌ Sorry, you are not eligible for a loan.")
    except Exception as e:
        st.error(f"Prediction failed due to: {e}")

    st.write("""
    We used a machine learning (Random Forest) model to predict your eligibility. 
    The features used in this prediction are ranked by relative importance below.
    """)
    st.image("feature_importance.png")
