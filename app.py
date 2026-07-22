import pickle
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# 1. Load the trained Random Forest model from GitHub
@st.cache_data
def load_model():
    url = "https://raw.githubusercontent.com/tariqul-tuhin/Thesis/main/random_forest_model.pkl"
    response = requests.get(url)
    if response.status_code == 200:
        return pickle.load(BytesIO(response.content))
    else:
        raise FileNotFoundError(f"Failed to download the model file. Status code: {response.status_code}")

classifier = load_model()

@st.cache_data
def prediction(gender, married, dependents, education, self_employed,
               applicant_income, coapplicant_income, loan_amount,
               loan_term, credit_history, property_area):

    # Map user-friendly text selections to matching numerical preprocessed values
    gender_val = 1 if gender == "Male" else 0
    married_val = 1 if married == "Yes" else 0

    if dependents == "3+":
        dependents_val = 3
    else:
        dependents_val = int(dependents)

    education_val = 0 if education == "Graduate" else 1
    self_employed_val = 1 if self_employed == "Yes" else 0

    credit_history_val = 1.0 if credit_history == "Clear Debts (Good)" else 0.0

    # Map Property Area back to its encoded floats observed in your dataset
    if property_area == "Urban":
        property_val = 0.6584158415841584
    elif property_area == "Semi-Urban":
        property_val = 0.7682403433476395
    else: # Rural
        property_val = 0.6145251396648045

    # 2. Build input dataframe matching exact training columns
    input_data = pd.DataFrame(
        [[gender_val, married_val, dependents_val, education_val, self_employed_val,
          applicant_income, coapplicant_income, loan_amount, loan_term,
          credit_history_val, property_val]],
        columns=[
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
            "Credit_History", "Property_Area"
        ]
    )

    # Force column alignment to prevent shape mismatch errors
    input_data = input_data[classifier.feature_names_in_]

    # 3. Predict (0 = Rejected, 1 = Approved)
    pred = classifier.predict(input_data)
    return "Approved" if pred[0] == 1 else "Rejected"

def main():
    # Page Banner
    st.markdown(
        """
        <div style="background-color:#F4D03F;padding:13px;border-radius:10px;margin-bottom:20px">
        <h1 style="color:black;text-align:center;font-family:sans-serif;">Loan Prediction App (Random Forest)</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📋 Enter Applicant Details")

    # Layout inputs cleanly using columns
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox('Gender', ("Male", "Female"))
        Married = st.selectbox('Married Status', ("Yes", "No"))
        Dependents = st.selectbox('Number of Dependents', ("0", "1", "2", "3+"))
        Education = st.selectbox('Education', ("Graduate", "Under_Graduate"))
        Self_Employed = st.selectbox('Self Employed Status', ("No", "Yes"))
        Property_Area = st.selectbox('Property Area Type', ("Urban", "Semi-Urban", "Rural"))

    with col2:
        ApplicantIncome = st.number_input("Applicant's Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
        CoapplicantIncome = st.number_input("Co-applicant's Monthly Income ($)", min_value=0.0, value=0.0, step=100.0)
        LoanAmount = st.number_input("Requested Loan Amount (in thousands, e.g., 128 = $128,000)", min_value=0.0, value=128.0, step=1.0)
        Loan_Amount_Term = st.number_input("Loan Term Duration (in months)", min_value=0.0, value=360.0, step=12.0)
        Credit_History = st.selectbox("Credit History Assessment", ("Clear Debts (Good)", "Unclear Debts (Bad)"))

    st.write("---")

    # Prediction Action Button
    if st.button("Submit Application for Review"):
        with st.spinner("Running eligibility checks against Random Forest matrix..."):
            result = prediction(
                Gender, Married, Dependents, Education, Self_Employed,
                ApplicantIncome, CoapplicantIncome, LoanAmount,
                Loan_Amount_Term, Credit_History, Property_Area
            )

        # Output blocks without SHAP visuals
        if result == "Approved":
            st.success(f'Decision outcome: **Your loan application is {result}!**', icon="✅")
        else:
            st.error(f'Decision outcome: **Your loan application is {result}.**', icon="❌")

if __name__ == '__main__':
    main()
