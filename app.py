import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Geely Auto Fuel Type Prediction", layout="wide")

model = pickle.load(open("pkl/fueltype_model.pkl", "rb"))
scaler = pickle.load(open("pkl/scaler.pkl", "rb"))
feature_columns = pickle.load(open("pkl/feature_columns.pkl", "rb"))

st.title("🚗 Geely Auto - US Market Fuel Type Prediction")
st.write("Predict whether a car should use Gas or Diesel based on specifications.")

st.sidebar.header("Enter Car Specifications")

enginesize = st.sidebar.number_input("Engine Size", 50, 400, 120)
compressionratio = st.sidebar.number_input("Compression Ratio", 5.0, 25.0, 10.0)
horsepower = st.sidebar.number_input("Horsepower", 40, 300, 100)
citympg = st.sidebar.number_input("City MPG", 5, 60, 25)
highwaympg = st.sidebar.number_input("Highway MPG", 10, 70, 30)
price = st.sidebar.number_input("Price", 5000, 50000, 15000)
curbweight = st.sidebar.number_input("Curb Weight", 1000, 5000, 2500)
stroke = st.sidebar.number_input("Stroke", 2.0, 5.0, 3.0)
carlength = st.sidebar.number_input("Car Length", 120.0, 220.0, 170.0)
carwidth = st.sidebar.number_input("Car Width", 50.0, 80.0, 65.0)

input_data = pd.DataFrame({
    'enginesize': [enginesize],
    'compressionratio': [compressionratio],
    'horsepower': [horsepower],
    'citympg': [citympg],
    'highwaympg': [highwaympg],
    'price': [price],
    'curbweight': [curbweight],
    'stroke': [stroke],
    'carlength': [carlength],
    'carwidth': [carwidth]
})

input_data = input_data[feature_columns]
input_scaled = scaler.transform(input_data)

if st.sidebar.button("Predict Fuel Type"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🚗 Recommended Fuel Type: DIESEL")
    else:
        st.success("🚗 Recommended Fuel Type: GAS")

    st.write("### Probability")
    st.write(f"Gas: {round(probability[0]*100,2)}%")
    st.write(f"Diesel: {round(probability[1]*100,2)}%")

    st.info("Business Insight: Diesel vehicles are typically preferred when engine size and compression ratio are higher along with strong fuel efficiency metrics.")