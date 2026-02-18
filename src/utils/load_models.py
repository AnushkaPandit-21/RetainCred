import joblib
import os

CHURN_MODEL_PATH = "data/processed/churn_model.pkl"
KMEANS_PATH = "data/processed/kmeans_model.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
FEATURE_COLUMNS_PATH = "data/processed/feature_columns.pkl"
SEG_SCALER_PATH = "data/processed/segmentation_scaler.pkl"
SEG_FEATURES_PATH = "data/processed/seg_features.pkl"


def load_models():
    paths = [CHURN_MODEL_PATH, KMEANS_PATH, SCALER_PATH, SEG_SCALER_PATH, FEATURE_COLUMNS_PATH, SEG_FEATURES_PATH]
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found at: {path}")

    churn_model = joblib.load(CHURN_MODEL_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    segmentation_scaler = joblib.load(SEG_SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    seg_features = joblib.load(SEG_FEATURES_PATH)

    return churn_model, kmeans, scaler, segmentation_scaler, feature_columns, seg_features
