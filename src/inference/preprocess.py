import pandas as pd

def preprocess_input(df, feature_columns):

    # --- Feature engineering (same as notebook) ---
    df['avg_spend_per_txn'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']
    df['credit_used_ratio'] = df['Total_Revolving_Bal'] / df['Credit_Limit']
    df['tenure_years'] = df['Months_on_book'] / 12

    df['low_txn_flag'] = (df['Total_Trans_Ct'] < 50).astype(int)
    df['high_inactive_flag'] = (df['Months_Inactive_12_mon'] >= 3).astype(int)
    df['high_contact_flag'] = (df['Contacts_Count_12_mon'] >= 3).astype(int)

    # One-hot encode (will create fewer columns than training)
    df = pd.get_dummies(df)

    # Add missing columns from training
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep correct order
    df = df[feature_columns]

    return df
