def preprocess_segmentation(df, seg_features):
    df_seg = df.copy()

    df_seg['avg_spend_per_txn'] = df_seg['Total_Trans_Amt'] / df_seg['Total_Trans_Ct']
    df_seg['credit_used_ratio'] = df_seg['Total_Revolving_Bal'] / df_seg['Credit_Limit']
    df_seg['tenure_years'] = df_seg['Months_on_book'] / 12

    return df_seg[seg_features]
