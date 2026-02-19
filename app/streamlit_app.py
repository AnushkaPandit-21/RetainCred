import streamlit as st
from src.inference.pipeline import full_prediction
import pandas as pd

st.title("💳 RetainCred — Churn Prediction & Offer Engine")

st.header("Enter Customer Details")

Customer_Age = st.number_input("Customer Age", 18, 90, 45)
Gender = st.selectbox("Gender", ["M","F"])
Dependent_count = st.number_input("Dependents", 0, 5, 2)
Education_Level = st.selectbox("Education", 
    ["Graduate","High School","College","Post-Graduate","Doctorate","Uneducated"])
Marital_Status = st.selectbox("Marital Status",
    ["Single","Married","Divorced"])
Income_Category = st.selectbox("Income",
    ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +"])
Card_Category = st.selectbox("Card Category",
    ["Blue","Silver","Gold","Platinum"])

Months_on_book = st.slider("Months on Book", 1, 60, 36)
Total_Relationship_Count = st.slider("Relationship Count", 1, 6, 4)
Months_Inactive_12_mon = st.slider("Inactive Months", 0, 6, 2)
Contacts_Count_12_mon = st.slider("Bank Contacts", 0, 6, 2)

Credit_Limit = st.number_input("Credit Limit", 1000, 50000, 12000)
Total_Revolving_Bal = st.number_input("Revolving Balance", 0, 20000, 500)
Avg_Open_To_Buy = st.number_input("Open To Buy", 0, 50000, 10000)

Total_Amt_Chng_Q4_Q1 = st.slider("Amount Change Q4→Q1", 0.0, 3.0, 1.0)
Total_Trans_Amt = st.number_input("Total Transaction Amount", 100, 20000, 2000)
Total_Trans_Ct = st.slider("Transaction Count", 1, 150, 50)
Total_Ct_Chng_Q4_Q1 = st.slider("Txn Count Change Q4→Q1", 0.0, 3.0, 1.0)
Avg_Utilization_Ratio = st.slider("Utilization Ratio", 0.0, 1.0, 0.3)

if st.button("Predict Customer Risk"):
    payload = {
        "Customer_Age":Customer_Age,
        "Gender":Gender,
        "Dependent_count":Dependent_count,
        "Education_Level":Education_Level,
        "Marital_Status":Marital_Status,
        "Income_Category":Income_Category,
        "Card_Category":Card_Category,
        "Months_on_book":Months_on_book,
        "Total_Relationship_Count":Total_Relationship_Count,
        "Months_Inactive_12_mon":Months_Inactive_12_mon,
        "Contacts_Count_12_mon":Contacts_Count_12_mon,
        "Credit_Limit":Credit_Limit,
        "Total_Revolving_Bal":Total_Revolving_Bal,
        "Avg_Open_To_Buy":Avg_Open_To_Buy,
        "Total_Amt_Chng_Q4_Q1":Total_Amt_Chng_Q4_Q1,
        "Total_Trans_Amt":Total_Trans_Amt,
        "Total_Trans_Ct":Total_Trans_Ct,
        "Total_Ct_Chng_Q4_Q1":Total_Ct_Chng_Q4_Q1,
        "Avg_Utilization_Ratio":Avg_Utilization_Ratio
    }

    df = pd.DataFrame([payload])
    result = full_prediction(df)

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
    st.success(f"Segment: {result['segment']}")
    st.warning(f"Risk Level: {result['risk_level']}")
    st.info(f"Recommended Offer: {result['offer']}")
    st.write("📩 Message:")
    st.write(result['message'])
