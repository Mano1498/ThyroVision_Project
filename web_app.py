# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:07:15 2024

@author: ranja
"""
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from joblib import load
import logging

# Configure logging
logging.basicConfig(filename='thyroid_detection.log', level=logging.INFO)

# Load the scaler and XGBoost model
scaler = load('std.joblib')
model = load('xgbcl.joblib')

# Function to preprocess input data
def preprocess_data(data):
    data['sex'] = data['sex'].map({'female': 0, 'male': 1})
    return data

# Function to make predictions
def predict_thyroid_status(data):
    # Preprocess input data
    data = preprocess_data(data)

    # Ensure the order of features matches the order during training
    feature_order = ['FTI', 'T3', 'T4U', 'TSH', 'TT4', 'age', 'on_thyroxine', 'sex']

    # Reorder columns to match the order during training
    data = data[feature_order]

    # Standardize the input data using the loaded scaler
    scaled_data = scaler.transform(data)

    # Make predictions using the XGBoost model
    try:
        predictions = model.predict(scaled_data)
        logging.info("Prediction successful")
        return predictions
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit app
def main():

    st.title("Thyroid Detection")

    # Sidebar for user input
    st.header("User Input")

    # Get user input for thyroid features
    age = st.slider("Age", 1, 100, 25)
    sex = st.radio("Sex", ['female', 'male'])
    fti = st.number_input("FTI")
    t3 = st.number_input("T3")
    t4u = st.number_input("T4U")
    tsh = st.number_input("TSH")
    tt4 = st.number_input("TT4")
    on_thyroxine = st.checkbox("On Thyroxine")

    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'FTI': [fti],
        'T3': [t3],
        'T4U': [t4u],
        'TSH': [tsh],
        'TT4': [tt4],
        'age': [age],
        'on_thyroxine': [on_thyroxine],
        'sex': [sex],
    })

    # Make predictions (make sure to define this function)
    result = predict_thyroid_status(input_data)

    # Display the result
    st.subheader("Prediction")
    if result is not None:
        if result[0] == 0:
            st.success("Normal")
        elif result[0] == 1:
            st.warning("Hypothyroid")
        else:
            st.error("Hyperthyroid")

# Run the app
if __name__ == '__main__':
    main()


import streamlit as st
import pandas as pd
from sklearn import __version__ as sklearn_version
from xgboost import __version__ as xgboost_version
from joblib import __version__ as joblib_version

# Display library versions using print
print(f"Streamlit version: {st.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn_version}")
print(f"XGBoost version: {xgboost_version}")
print(f"Joblib version: {joblib_version}")
