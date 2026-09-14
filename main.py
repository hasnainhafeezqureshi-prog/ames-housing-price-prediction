import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl')
defaults = joblib.load('defaults.pkl')
columns = joblib.load('columns.pkl')

app = FastAPI()

class InputData(BaseModel):
    OverallQual: int
    TotalSF: float
    GarageCars: int
    FullBath: int
    HouseAge: int
    RemodAge: int
    LotArea: float
    BsmtFullBath: int
    TotRmsAbvGrd: int
    Fireplaces: int

@app.post("/predict")
def predict_price(data: InputData):
    input_dict = defaults.copy()
    input_dict.update(data.model_dump())
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[columns]
    scaled_input = scaler.transform(input_df)
    prediction_log = model.predict(scaled_input)
    predicted_price = np.exp(prediction_log[0])  # Convert log price back to original scale
    print(f"Predicted price: {predicted_price}")
    return {'predicted_price': float(predicted_price)}
