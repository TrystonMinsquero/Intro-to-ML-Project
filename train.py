import streamlit as st
from data_cleaner import *
import pandas as pd
from data_visualizer import fetch_and_clean_data
from data_uploader import get_dataset_names
from model_design import get_model, get_model_names, save_model_as
# Importing required libraries
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential, load_model, save_model
from sklearn.model_selection import train_test_split 
from os import listdir
from os.path import isfile, join
from keras import callbacks

def app():
    st.title("Training")

    if(len(get_model_names()) <= 0):
        st.write("No models to train! Create a model in model design.")
        return

    if(len(get_dataset_names()) <= 0):
        st.write("No datasets to train on! Create a dataset in data uploader.")
        return

    modelname = st.radio('Choose Model to Train', get_model_names())

    dataset_name = st.radio("Select Dataset", get_dataset_names())

    data_df = fetch_and_clean_data(dataset_name).copy()
    data_df = add_sentiments(data_df)
    X = vectorize_data(data_df)

    model = get_model(modelname)
    
    # model = create_model(X)
    # Get inputs
    helpLink = "https://keras.io/getting_started/faq/#what-do-sample-batch-and-epoch-mean"

    form = st.form(key="Test_parameters")
    form.title("Training Parameters")

    test_size_percent = form.number_input("Validation Percentage", 0.0, 1.0, value=.3,
        help="will train on 1 - validation percentage, and evaluate model later")

    batch_size = form.number_input("Batch Size", 1, len(data_df), value=100,
        help=f"See {helpLink} for more info about the variables")

    epochs = form.number_input("Epochs", 1, 
        help=f"See {helpLink} for more info about the variables")

    submit = form.form_submit_button("Start Training")

    if submit:
        
        #Splitting the data into training and testing
        y=pd.get_dummies(data_df['sentiment'])
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size_percent)
        stepsPerEpoch = int( int((X_train.shape[0] / batch_size)+1))
        model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, callbacks = [TrainCallback(epochs, stepsPerEpoch)])

        complete = st.text("Validating Model...")
        loss, accuracy = model.evaluate(X_test, y_test)
        complete.text(f"Loss: {round(loss,4)}   and   Accuracy: {round(accuracy, 4)}")

        # Test to make sure the model works
        # singlePrediction = "This a an amazing product. I love it and it's great"
        # cleaned_text = clean_text(singlePrediction)
        # st.text(cleaned_text)
        # X = vectorize_text(cleaned_text)
        # st.write(f'prediction: {model(X)}')

        save_model_as(modelname, model)
        st.success('Model saved to ' + join('models', modelname + '.keras'))
        print('Model saved to ' + join('models', modelname + '.keras'))


class TrainCallback(callbacks.Callback):
    
    def __init__(self, epochNum, stepsPerEpoch) -> None:
        super().__init__()
        self.epochNum = epochNum
        self.stepsPerEpoch = stepsPerEpoch
        self.epochLabel = st.text("Epoch 0/" + str(epochNum))
        self.epochBar = st.progress(0.0)
        self.batchLabel = st.text("Batch 0/" + str(stepsPerEpoch))
        self.batchBar = st.progress(0.0)

    def on_train_begin(self, logs=None):
        print("Start training")

    def on_train_end(self, logs=None):
        print("Stop training")

    def on_epoch_begin(self, epoch, logs=None):
        self.epochLabel.text(f"Epoch {epoch + 1}/{(self.epochNum)}")
        self.batchBar.progress(0.0)

    def on_batch_begin(self, batch, logs=None):
        self.batchLabel.text(f"Batch {batch + 1}/{(self.stepsPerEpoch)}")

    def on_epoch_end(self, epoch, logs=None):
        self.epochBar.progress((epoch + 1.0)/(self.epochNum))

    def on_train_batch_end(self, batch, logs=None):
        self.batchBar.progress((batch + 1.0)/(self.stepsPerEpoch))
