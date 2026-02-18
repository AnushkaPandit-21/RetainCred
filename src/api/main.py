from fastapi import FastAPI
import pandas as pd
from src.inference.pipeline import full_prediction

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RetainCred API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    result = full_prediction(df)
    return result
