# 💳 RetainCred

### AI-Powered Credit Card Churn Prediction, Segmentation & Personalized Offer Engine

🔗 **Live App:**
[https://retaincred-app-myfirstproject21042004.streamlit.app/](https://retaincred-app-myfirstproject21042004.streamlit.app/)

---

## Overview

RetainCred is an end-to-end machine learning product that predicts credit-card customer churn, segments users by behaviour, and generates personalized retention offers using AI.

The system combines:

* Predictive modelling
* Customer segmentation
* Recommendation engine
* LLM-generated marketing messages
* Full deployment pipeline

---

## Problem Statement

Customer churn is one of the biggest revenue losses for credit card companies.

Instead of reacting after churn happens, RetainCred answers:

* Which customers are likely to churn?
* What type of users are they?
* What retention offer should we give?
* How do we communicate it personally?

---

## System Architecture

```
User Input → Feature Engineering → Churn Model (XGBoost)
                                 → Segmentation (KMeans)
                                 → Offer Recommendation
                                 → LLM Message Generation
                                 → Streamlit App
```

---

## Dataset

Kaggle Credit Card Customers Dataset
~10,000 customers with demographic, behavioural and transaction data.

---

## Machine Learning Pipeline

### 1️. Churn Prediction Model

Models evaluated:

* Logistic Regression
* Random Forest
* **XGBoost (final model)**

Cross-validation results:

| Model               | ROC-AUC   | Recall   |
| ------------------- | --------- | -------- |
| Logistic Regression | 0.93      | 0.63     |
| Random Forest       | 0.99      | 0.81     |
| **XGBoost**         | **0.993** | **0.86** |

Final performance:

* ROC-AUC: **0.993**
* F1 Score: **0.91**
* Recall (Churn detection): **86%**

Threshold tuning performed to optimise recall–precision trade-off.

---

### 2️. Customer Segmentation

KMeans clustering using behavioural features:

Features used:

* Transaction count & amount
* Credit utilisation
* Relationship count
* Engagement metrics
* Derived features (avg spend/txn, credit used ratio)

Segments discovered:

* **High Value Spenders**
* **Credit Revolvers**
* **Low Engagement Users**

---

### 3️. Risk Tier Classification

Churn probability is converted into:

* Low Risk
* Medium Risk
* High Risk

This allows business-friendly decision making.

---

### 4️. Offer Recommendation Engine

Offers generated based on:

* Risk tier
* Spending behaviour
* Engagement level
* Credit utilisation

Examples:

* Fee waivers for high-risk users
* Cashback offers for active users
* Reward boosters for loyal customers

---

### 5️. LLM Message Generation

HuggingFace Inference API (FLAN-T5) generates personalised messages:

Example output:

> “We value your loyalty! Enjoy bonus reward points on your next purchases.”

---

## Production Engineering

### Backend

* FastAPI inference service
* Modular pipeline architecture
* Separate preprocessing pipelines
* Saved model artifacts & scalers

### Frontend

* Streamlit interactive dashboard
* Real-time predictions

### DevOps

* Docker containerisation
* GitHub Actions CI pipeline
* Environment variable secret management
* Cloud deployment (Streamlit Community Cloud)

---

## Run locally with Docker

```bash
docker build -t retaincred .
docker run -e HF_TOKEN=your_token -p 8501:8501 retaincred
```

---


