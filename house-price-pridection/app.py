import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

model = pickle.load(open("model.pkl", "rb"))

st.markdown("<h1 style='text-align:center;'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, step=100)
    bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])

with col2:
    bathrooms = st.selectbox("Bathrooms", [1,2,3,4])

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Estimated Price: ₹ {prediction[0]:,.0f}")
    st.balloons()

if st.checkbox("Show Dataset"):
    df = pd.read_csv("housing.csv")
    st.write(df.head())

st.markdown("---")
st.caption("Made by Gajendra 🚀")
