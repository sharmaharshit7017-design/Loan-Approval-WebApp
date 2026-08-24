# 🏦 Loan Approval Prediction — Web App

A Machine Learning web application that predicts whether a loan application is likely to be approved based on the applicant's financial and personal information.

The application is built with **Streamlit** and uses a trained Scikit-learn machine learning pipeline for predictions.

## 🚀 Live Demo

👉 **[Open the Loan Approval Prediction App](https://loan-approval-webapp-ovdv83qerdrmfbgd2rhdet.streamlit.app/)**

---

## 📌 About the Project

This web application provides a simple interface where users can enter an applicant's details and receive a machine learning prediction for loan approval.

The application uses the trained model from the **Loan Approval Prediction** machine learning project.

### Prediction

- ✅ **Approved**
- ❌ **Rejected**

---

## 🧾 Input Features

The application accepts the following information:

### Personal Information

- Gender
- Married
- Dependents
- Education
- Self Employed

### Financial Information

- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History

### Property Information

- Property Area

---

## 🤖 Machine Learning Model

The web application uses a trained **Logistic Regression** model.

The final model achieved:

- **Accuracy:** 86.18%
- **ROC-AUC:** 0.852

The model was selected after comparing:

- Logistic Regression
- Decision Tree
- Random Forest

The complete preprocessing and machine learning pipeline is stored in:

```text
loan_model.pkl