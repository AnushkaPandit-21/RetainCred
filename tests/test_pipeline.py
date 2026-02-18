import pandas as pd
import json
import sys
import os

# Ensure src is in python path if running directly
sys.path.append(os.getcwd())

from src.inference.pipeline import full_prediction

def test_pipeline():
    # Create a dummy input dataframe
    # We need features expected by preprocess_input (from feature_columns)
    # and preprocess_segmentation.
    
    data = {
        'Customer_Age': [45],
        'Gender': ['M'],
        'Dependent_count': [3],
        'Education_Level': ['Graduate'],
        'Marital_Status': ['Married'],
        'Income_Category': ['$60K - $80K'],
        'Card_Category': ['Blue'],
        'Months_on_book': [39],
        'Total_Relationship_Count': [5],
        'Months_Inactive_12_mon': [1],
        'Contacts_Count_12_mon': [3],
        'Credit_Limit': [12691.0],
        'Total_Revolving_Bal': [777],
        'Avg_Open_To_Buy': [11914.0],
        'Total_Amt_Chng_Q4_Q1': [1.335],
        'Total_Trans_Amt': [1144],
        'Total_Trans_Ct': [42],
        'Total_Ct_Chng_Q4_Q1': [1.625],
        'Avg_Utilization_Ratio': [0.061]
    }
    
    df = pd.DataFrame(data)
    
    print("Running full_prediction...")
    try:
        result = full_prediction(df)
        print("Prediction successful!")
        print(json.dumps(result, indent=2))
        
        # Validation checks
        assert "churn_probability" in result
        assert "segment" in result
        assert "risk_level" in result
        assert "offer" in result
        assert "message" in result
        
        print("\nRisk Level:", result['risk_level'])
        print("Segment:", result['segment'])
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
