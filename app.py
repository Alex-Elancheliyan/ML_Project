import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict car price
def predict_price(year, present_price, kms_driven, owner, fuel_diesel, fuel_petrol, seller_individual, transmission_manual):
    input_data = {'Year': year,
                  'Present_Price': present_price,
                  'Kms_Driven': kms_driven,
                  'Owner': owner,
                  'Fuel_Type_Diesel': fuel_diesel,
                  'Fuel_Type_Petrol': fuel_petrol,
                  'Seller_Type_Individual': seller_individual,
                  'Transmission_Manual': transmission_manual}
    input_df = pd.DataFrame(input_data, index=[0])
    predicted_price = model.predict(input_df)[0]
    return predicted_price

# Streamlit UI
st.title('Car Price Preditor:')

# Sidebar inputs
st.sidebar.header('Enter Car Details')
year = st.sidebar.number_input('Year of Manufacture', min_value=1990, max_value=2024, value=2010)
present_price = st.sidebar.number_input('Present Price (in lakhs)', min_value=0.5, max_value=50.0, value=5.0)
kms_driven = st.sidebar.number_input('Kilometers Driven', min_value=500, max_value=500000, value=50000)
owner = st.sidebar.selectbox('Number of Previous Owners', [0, 1, 2, 3])
fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel'])
seller_type = st.sidebar.selectbox('Seller Type', ['Individual', 'Dealer'])
transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])

fuel_diesel = 1 if fuel_type == 'Diesel' else 0
fuel_petrol = 1 if fuel_type == 'Petrol' else 0
seller_individual = 1 if seller_type == 'Individual' else 0
transmission_manual = 1 if transmission == 'Manual' else 0

# Prediction and display
if st.sidebar.button('Predict'):
    predicted_price = predict_price(year, present_price, kms_driven, owner, fuel_diesel, fuel_petrol, seller_individual, transmission_manual)
    st.success(f'Your Predicted Car Price: {predicted_price:.2f} lakhs')
