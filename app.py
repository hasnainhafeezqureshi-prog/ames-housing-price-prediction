import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl')
defaults = joblib.load('defaults.pkl')
columns = joblib.load('columns.pkl')

st.title("Ames Housing Price Predictor")

overall_qual = st.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
total_sf = st.number_input("Total Square Footage", min_value=0, value=1500)
garage_cars = st.number_input("Garage Capacity (cars)", min_value=0, max_value=4, value=2)
full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=2)
house_age = st.number_input("House Age (years)", min_value=0, value=20)
remod_age = st.number_input("Years Since Remodel", min_value=0, value=15)
lot_area = st.number_input("Lot Area (sqft)", min_value=0, value=8000)
bsmt_full_bath = st.number_input("Basement Full Bathrooms", min_value=0, max_value=3, value=1)
tot_rooms = st.number_input("Total Rooms Above Ground", min_value=1, value=6)
fireplaces = st.number_input("Number of Fireplaces", min_value=0, max_value=4, value=1)


if st.button("Predict Price"):
    input_dict = defaults.copy()
    input_dict.update({
        'OverallQual': overall_qual,
        'TotalSF': total_sf,
        'GarageCars': garage_cars,
        'FullBath': full_bath,
        'HouseAge': house_age,
        'RemodAge': remod_age,
        'LotArea': lot_area,
        'BsmtFullBath': bsmt_full_bath,
        'TotRmsAbvGrd': tot_rooms,
        'Fireplaces': fireplaces
    })

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[columns]
    scaled_input = scaler.transform(input_df)
    prediction_log = model.predict(scaled_input)
    predicted_price = np.exp(prediction_log[0])

    st.success(f"Predicted Price: ${predicted_price:,.2f}")