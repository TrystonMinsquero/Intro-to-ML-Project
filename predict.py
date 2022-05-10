import streamlit as st
from data_cleaner import *
import pandas as pd
from data_visualizer import fetch_and_clean_data
from model_design import get_model
import redirect as rd
# Importing required libraries
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split 
from os import listdir
from os.path import isfile, join
import numpy as np

def app():
    st.title("Prediction")

    if(len(get_model_names()) <= 0):
        st.write("No models to train! Create a model in model design.")
        return

    form = st.form(key="Test_parameters")

    modelname = form.radio('Choose Model to Predict with', get_model_names())

    
    singlePrediction = form.text_area("Sample Review")
    
    submit = form.form_submit_button("Predict")


    if submit:

        model = get_model(modelname)
        print(model.summary())

        if singlePrediction:      
            st.write("Cleaned Text")
            cleaned_text = clean_text(singlePrediction)
            st.text(cleaned_text)
            X = vectorize_text(cleaned_text)
            # st.write(f'input: {X}')
            st.write(f'prediction: {model(X)}')
            
        
        

def get_model_names():
    onlyfiles = [f for f in listdir('models') if isfile(join('models', f))]
    model_names = []
    for file in onlyfiles:
        model_names.append(file.removesuffix('.keras'))
    return model_names
