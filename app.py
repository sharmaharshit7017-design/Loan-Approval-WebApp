import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ==========================
# Load Files
# ==========================

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==========================
# Title
# ==========================

st.title("🏦 Loan Approval Prediction")
st.write("Fill the applicant details below.")

# ==========================
# Input Form
# ==========================

with st.form("loan_form"):

    gender = st.selectbox("Gender", ["Male", "Female"])

    married = st.selectbox("Married", ["Yes", "No"])

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

    applicantincome = st.number_input(
        "Applicant Income",
        min_value=0
    )

    coapplicantincome = st.number_input(
        "Co Applicant Income",
        min_value=0
    )

    loanamount = st.number_input(
        "Loan Amount",
        min_value=0
    )

    loan_amount_term = st.number_input(
        "Loan Amount Term",
        min_value=0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

    submit = st.form_submit_button("Predict")

# ==========================
# Prediction
# ==========================

if submit:

    sample = pd.DataFrame({

        "gender":[gender],
        "married":[married],
        "dependents":[dependents],
        "education":[education],
        "self_employed":[self_employed],
        "applicantincome":[applicantincome],
        "coapplicantincome":[coapplicantincome],
        "loanamount":[loanamount],
        "loan_amount_term":[loan_amount_term],
        "credit_history":[credit_history],
        "property_area":[property_area]

    })

    # One Hot Encoding
    sample = pd.get_dummies(sample, drop_first=True)

    # Match Training Columns
    sample = sample.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale Data
    sample = scaler.transform(sample)

    # Prediction
    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0][1]

    st.divider()

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric(
        "Approval Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(float(probability))