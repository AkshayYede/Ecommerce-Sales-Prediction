import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="E-Commerce Sales Predictor", layout="wide")

# Load model and scaler
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/model.pkl')

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        h1 {
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .stNumberInput label {
            font-size: 1.1rem;
        }

        .stButton button {
            width: 100%;
            padding: 0.5rem 1rem;
            font-size: 1.1rem;
        }

        .prediction-result {
            font-size: 1.3rem;
            font-weight: 600;
            color: green;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title section centered
st.markdown("<h1>E-Commerce Sales Predictor</h1>", unsafe_allow_html=True)

# Input fields centered using columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    avg_session_length = st.number_input(
        "On average, how long is your browsing session? (in minutes)",
        min_value=0.0, value=30.0, step=1.0
    )

    time_on_app = st.number_input(
        "How much time do you spend on the app daily? (in minutes)",
        min_value=0.0, value=12.0, step=1.0
    )

    length_of_membership = st.number_input(
        "How many years have you been a member?",
        min_value=0.0, value=4.0, step=0.1
    )

    if st.button("Predict My Spending"):
        input_data = np.array([avg_session_length, time_on_app, length_of_membership]).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)

        st.markdown(f"<div class='prediction-result'>Your estimated yearly amount spent is: <strong>${prediction[0]:.2f}</strong></div>", unsafe_allow_html=True)
