import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def rebuild_artifacts():
    print("Loading data...")
    # Load full processed data for scaler fitting
    # Note: notebook 04 used X_cluster which was derived from original data. 
    # full_processed_data.csv seems to be the best source available.
    df = pd.read_csv('data/processed/full_processed_data.csv')
    
    # Load segmentation features list
    seg_features = joblib.load('data/processed/seg_features.pkl')
    print(f"Segmentation features: {seg_features}")
    
    # Create and fit scaler for segmentation features
    print("Fitting segmentation scaler...")
    seg_scaler = StandardScaler()
    # Check if features exist in df
    missing_cols = [col for col in seg_features if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in full_processed_data: {missing_cols}")
        # Try to reconstruct them if possible or error out
        # In preprocess_segmentation.py, features are created. 
        # But we need statistics from the TRAINING/FULL distribution.
        # Let's check if we can calculate them from customer_intelligence.csv if full_processed_data is missing them.
    
    X_seg = df[seg_features]
    seg_scaler.fit(X_seg)
    
    # Save the scaler
    joblib.dump(seg_scaler, 'data/processed/segmentation_scaler.pkl')
    print("Saved data/processed/segmentation_scaler.pkl")
    
    # Calculate Spend and Transaction Thresholds (Terciles)
    # Notebook 05 uses pd.qcut(df['Total_Trans_Amt'], 3, labels=['Low', 'Medium', 'High'])
    print("\nCalculating Spend/Txn Thresholds...")
    
    # We need the original distribution. full_processed_data might have scaled/processed features?
    # No, it should have original columns too if specific steps weren't taken to drop them. 
    # Let's check columns. If not, use 'customer_intelligence.csv'
    
    try:
        # Try customer_intelligence.csv for raw values
        raw_df = pd.read_csv('data/processed/customer_intelligence.csv')
        print("Using customer_intelligence.csv for thresholds")
        target_df = raw_df
    except:
        print("Using full_processed_data.csv for thresholds")
        target_df = df
        
    spending = target_df['Total_Trans_Amt']
    txns = target_df['Total_Trans_Ct']
    
    # Terciles: 33.3% and 66.6%
    spend_low_cut = spending.quantile(1/3)
    spend_high_cut = spending.quantile(2/3)
    
    txn_low_cut = txns.quantile(1/3)
    txn_high_cut = txns.quantile(2/3)
    
    print(f"\nSPEND_THRESHOLD_LOW = {spend_low_cut:.2f}")
    print(f"SPEND_THRESHOLD_HIGH = {spend_high_cut:.2f}")
    print(f"TXN_THRESHOLD_LOW = {txn_low_cut:.2f}")
    print(f"TXN_THRESHOLD_HIGH = {txn_high_cut:.2f}")

if __name__ == "__main__":
    rebuild_artifacts()
