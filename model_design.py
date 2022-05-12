import streamlit as st
from data_cleaner import *
# Importing required libraries
from collections import Counter
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from os.path import join, isfile
from os import listdir

def app():
    st.title("Model Design")
    max_words = 500

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
            model.compile(loss = 'binary_crossentropy' if category_num == 2 else 'categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

            model.save(join('models', name + '.keras'))
            st.success(f"Model saved to {join('models', name + '.keras')}")
            print(f"Model saved to {join('models', name + '.keras')}")


def get_model_names():
    onlyfiles = [f for f in listdir('models') if isfile(join('models', f))]
    model_names = []
    for file in onlyfiles:
        model_names.append(file.removesuffix('.keras'))
    return model_names

def get_model(name):
    return load_model(join('models', name + '.keras'))

def save_model_as(name, model):
    save_model(model, join('models', name + '.keras'))
    return join('models', name + '.keras')




# Create LTSM Model based on:
# https://www.analyticsvidhya.com/blog/2021/06/natural-language-processing-sentiment-analysis-using-lstm/
# @st.cache(persist=True, allow_output_mutation=True)
# def create_model(X):
#     model = Sequential()
#     model.add(Embedding(500, 120, input_length = X.shape[1]))
#     model.add(SpatialDropout1D(0.4))
#     model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(2,activation='softmax'))
#     model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
#     print(model.summary())
#     return model



