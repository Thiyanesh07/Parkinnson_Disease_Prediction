import streamlit as st
import pandas as pd
import numpy as np
import pickle 

    
with open('best_model.pkl','rb') as fp:
        model=pickle.load(fp)

with open('features_list.pkl','rb') as fp:
        selected_features=pickle.load(fp)
        



df = pd.read_csv('Parkinsson disease.csv')


st.set_page_config(page_title="Parkinson's Prediction", layout="wide")

st.title("Parkinson's Disease Prediction 🧠")
st.write("This app uses a machine learning model to predict the likelihood of Parkinson's disease based on vocal measurements. Please enter the patient's data in the input boxes below.")


st.subheader("Patient's Vocal Measurements")


user_inputs = {}


num_columns = 4
cols = st.columns(num_columns)


for i, feature in enumerate(selected_features):
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    
    
    with cols[i % num_columns]:
        user_inputs[feature] = st.number_input(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=None,
            step=0.001,
            format="%.gf", 
            placeholder="Enter value...."
        )


if st.button("Predict"):
    if None in user_inputs.values():
        st.error("Please fill in all the input fields before making a prediction.")
    else:
        input_df = pd.DataFrame([user_inputs])
        input_df = input_df[selected_features]
        
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.warning(f"The model predicts a high likelihood of Parkinson's Disease.", icon="⚠️")
            st.info(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(f"The model predicts a low likelihood of Parkinson's Disease.", icon="✅")
            st.info(f"Confidence: {prediction_proba[0][0]*100:.2f}%")