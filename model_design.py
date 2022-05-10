import streamlit as st
from torch import dropout
from data_loader import get_amazon_alexa_data
from data_cleaner import *
import pandas as pd
from data_manipulation import fetch_and_clean_data
import redirect as rd
# Importing required libraries
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
from os.path import join

def app():
    st.title("Model Design")
    max_words = 5000

    with st.form("Create Model", True):

        name = st.text_input("Name of model file", value='model', help='filename of the model, overwrites the model if already exists')
        num_words = st.number_input(label="Max words", min_value=1, value=max_words)
        output_dim = st.number_input(label='Output dimensions', min_value=1, value=15)
        hidden_units = st.number_input(label='Hidden units for the LSTM Layer', min_value=1, value=176)        
        dropout = st.number_input(label='Dropout rate for the LSTM layer', min_value=0.0, max_value=1.0, step=.05, value=.2)
        category_num = st.number_input(label="Number of output categories", min_value=2, max_value=5)
        
        # todo: move down
        submit = st.form_submit_button("Create Model")

        if submit:
            model = Sequential()
            model.add(Embedding(num_words, output_dim))
            model.add(LSTM(hidden_units, dropout=dropout))
            model.add(Dense(category_num, activation= 'sigmoid' if category_num == 2 else 'softmax'))
            model.compile(loss = 'binary_crossentropy' if category_num == 2 else 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.save(join('models', name + '.keras'))
            st.success(f"Model saved to {join('models', name + '.keras')}")
            print(f"Model saved to {join('models', name + '.keras')}")
    


# Create LTSM Model based on:
# https://www.analyticsvidhya.com/blog/2021/06/natural-language-processing-sentiment-analysis-using-lstm/
@st.cache(persist=True, allow_output_mutation=True)
def create_model(X):
    model = Sequential()
    model.add(Embedding(500, 120, input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print(model.summary())
    return model



