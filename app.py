import streamlit as st
import pandas as pd
import joblib


# =========================================================
# 1. LOAD TRAINED MODEL
# =========================================================

model = joblib.load("loan_model.pkl")


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)


# =========================================================
# 3. TITLE
# =========================================================

st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant's details below to predict "
    "whether the loan is likely to be approved."
)


# =========================================================
# 4. USER INPUTS
# =========================================================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Co-applicant Income",
    min_value=0.0,
    value=0.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=150.0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0.0,
    value=360.0
)

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)


# =========================================================
# 5. CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame({
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
})


# =========================================================
# 6. PREDICTION
# =========================================================

if st.button("Predict Loan Approval"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]


    # =====================================================
    # 7. DISPLAY RESULT
    # =====================================================

    if prediction == 1:

        st.success(
            "✅ Loan is likely to be APPROVED"
        )

    else:

        st.error(
            "❌ Loan is likely to be REJECTED"
        )


    st.write(
        f"Approval Probability: "
        f"**{probability * 100:.2f}%**"
    )
