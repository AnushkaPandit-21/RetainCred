import pandas as pd
from src.utils.load_models import load_models
from src.inference.churn import predict_churn
from src.inference.segmentation import predict_segment
from src.inference.recommender import recommend_offer
from src.llm.generator import generate_offer_message
from src.inference.preprocess import preprocess_input
from src.inference.preprocess_segmentation import preprocess_segmentation


churn_model, kmeans_model, scaler, segmentation_scaler, feature_columns, seg_features = load_models()

# Quantile thresholds from notebook 05_recommender (calculated via rebuild_artifacts.py)
SPEND_LOW_CUT = 2567.0
SPEND_HIGH_CUT = 4475.67
TXN_LOW_CUT = 54.0
TXN_HIGH_CUT = 76.0

def risk_level(p):
    # Thresholds from 04_segmentation.ipynb (maximizing recall at 0.30)
    if p >= 0.30:
        return "High Risk"
    elif p >= 0.15: # Approximate medium threshold based on distribution
        return "Medium Risk"
    return "Low Risk"

def full_prediction(input_df):

    processed_df = preprocess_input(input_df, feature_columns)
    
    # Scale input for Churn Model (it was trained on scaled data)
    processed_scaled = scaler.transform(processed_df)
    churn_prob = predict_churn(churn_model, processed_scaled)[0]

    seg_input = preprocess_segmentation(input_df, seg_features)
    
    # Scale input for Segmentation (using specific segmentation scaler)
    seg_scaled = segmentation_scaler.transform(seg_input)
    segment = predict_segment(kmeans_model, seg_scaled)[0]


    risk = risk_level(churn_prob)

    # Spend Level (Terciles)
    amt = input_df['Total_Trans_Amt'].iloc[0]
    if amt <= SPEND_LOW_CUT:
        spend_level = "Low Spend"
    elif amt <= SPEND_HIGH_CUT:
        spend_level = "Medium Spend"
    else:
        spend_level = "High Spend"

    # Transaction Level (Terciles)
    ct = input_df['Total_Trans_Ct'].iloc[0]
    if ct <= TXN_LOW_CUT:
        txn_level = "Low Activity"
    elif ct <= TXN_HIGH_CUT:
        txn_level = "Medium Activity"
    else:
        txn_level = "High Activity"

    row = {
        'risk_level': risk,
        'segment': segment,
        'spend_level': spend_level,
        'txn_level': txn_level
    }
    offer = recommend_offer(row)
    message = generate_offer_message(segment, risk, offer)

    return {
        "churn_probability": float(churn_prob),
        "segment": segment,
        "risk_level": risk,
        "offer": offer,
        "message": message
    }
