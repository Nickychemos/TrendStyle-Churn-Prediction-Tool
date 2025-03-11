import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the model with caching
@st.cache_resource()
def load_model():
    return joblib.load('RandomForest.pkl')

st.cache_resource.clear()

# Website title and subtitle
st.title('Customer Churn Prediction Tool')
st.header('This tool helps TrendStyles identify customers who are likely to churn in the next 3 months.')

# Load the trained model
model = load_model()

# User Input Form
if model:
    st.subheader('Please enter the following details:')

# Purchase amount
Purchase_Amount_USD = st.number_input(
    "Purchase amount",
    min_value=20, max_value=100, value=20,
    help="Enter a value in USD"
)

# Previous Purchases
Previous_Purchases = st.number_input(
    "Previous Purchases",
    min_value=1, max_value=50, value=1,
    help="Enter a value between 1 and 50"
)

# Review Rating
Review_Rating = st.number_input(
    "Review Rating",
    min_value=1.0, max_value=5.0, value=1.0,
    help="Enter a value between 1.0 and 5.0"
)

# Payment Method
Payment_Method = st.selectbox(
    "Payment Method",
    options=[(0, 'Credit Card'), (1, 'Bank Transfer'), (2, 'Cash'), 
             (3, 'PayPal'), (4, 'Venmo'), (5, 'Debit Card')],
    format_func=lambda x: x[1], 
    help="Choose a payment method"
)
Payment_Method_Value = Payment_Method[0]

# Shipping Type
Shipping_Type = st.selectbox(
    "Shipping Type",
    options=[(0, 'Express'), (1, 'Free Shipping'), (2, 'Next Day Air'), 
             (3, 'Standard'), (4, '2-Day Shipping'), (5, 'Store Pickup')],
    format_func=lambda x: x[1],
    help="Choose a shipping type"
)
Shipping_Type_Value = Shipping_Type[0]

# Preferred Payment Method
Preferred_Payment_Method = st.selectbox(
    "Preferred Payment Method",
    options=[(0, 'Venmo'), (1, 'Cash'), (2, 'Credit Card'), 
             (3, 'PayPal'), (4, 'Bank Transfer'), (5, 'Debit Card')],
    format_func=lambda x: x[1],
    help="Choose a preferred payment method"
)
Preferred_Payment_Method_Value = Preferred_Payment_Method[0]

if st.button('Predict'):
    # Create DataFrame for model input
    input_data = pd.DataFrame([[Purchase_Amount_USD, Previous_Purchases, Review_Rating, 
                                Payment_Method_Value, Shipping_Type_Value, Preferred_Payment_Method_Value]],
                            columns=["Purchase_Amount_USD", "Previous_Purchases", "Review_Rating", 
                                    "Payment_Method", "Shipping_Type", "Preferred_Payment_Method"])

    # Predict Churn
    prediction = model.predict(input_data)

    # Churn Mapping
    performance_dict = {0: "Not Churn", 1: "Churn"}
    churn = performance_dict.get(prediction[0], "Unknown")

    # Display Results
    st.subheader(f'✅ This customer is likely to **{churn}** in the next 3 months')
st.write('Use this result to make informed decisions about customer retention.')
