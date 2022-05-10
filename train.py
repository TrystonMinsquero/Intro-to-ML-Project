import streamlit as st
from data_loader import get_amazon_alexa_data
from data_cleaner import *
import pandas as pd
from data_manipulation import fetch_and_clean_data
import redirect as rd
# Importing required libraries
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
from os import listdir
from os.path import isfile, join

def app():
    st.title("Training")

    data_df, X = fetch_and_clean_data()
    if(len(get_model_names()) <= 0):
        st.write("No models to train! Create a model in model design.")
        return

    modelname = st.radio('Choose Model to Train', get_model_names())

    model = load_model(join('models', modelname + '.keras'))
    
    # model = create_model(X)
    # Get inputs
    helpLink = "https://keras.io/getting_started/faq/#what-do-sample-batch-and-epoch-mean"

    form = st.form(key="Test_parameters")
    form.title("Training Parameters")

    test_size_percent = form.number_input("Validation Percentage", 0.0, 1.0, value=.3,
        help="will train on 1 - validation percentage, and evaluate model later")

    batch_size = form.number_input("Batch Size", 1, len(data_df),
        help=f"See {helpLink} for more info about the variables")

    epochs = form.number_input("Epochs", 1, 
        help=f"See {helpLink} for more info about the variables")

    submit = form.form_submit_button("Start Training")

    if submit:
        
        #Splitting the data into training and testing
        y=pd.get_dummies(data_df['sentiment'])
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size_percent, random_state = 42)
        with rd.stdout:
            model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, verbose = 'auto')
            loss, accuracy = model.evaluate(X_test, y_test)
            st.write(f"Loss: {round(loss,4)}   and   Accuracy: {round(accuracy, 4)}")
        save = st.button('Save Model')
        if save:
            model.save(join('models', modelname + '.keras'))


def get_model_names():
    onlyfiles = [f for f in listdir('models') if isfile(join('models', f))]
    model_names = []
    for file in onlyfiles:
        model_names.append(file.removesuffix('.keras'))
    return model_names


# Create LTSM Model 
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



