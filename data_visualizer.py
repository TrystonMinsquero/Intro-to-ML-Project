import streamlit as st
from data_cleaner import *
import pandas as pd
import matplotlib.pyplot as plt
from data_uploader import *
from model_design import get_model_names, get_model
from predict import add_predictions



def app():

    dataset_name = st.radio("Select Dataset", get_dataset_names())
    
    modelname = st.radio('Choose Model to Train', get_model_names())

    model = get_model(modelname)

    # get raw data
    data_df = get_dataset(dataset_name)
    # clean data 
    data_df = clean_data(data_df)
    # add sentiments based off rating
    data_df = add_sentiments(data_df)
    # fit labels to cleaned data
    data_df = fit_labels(data_df)
    # convert text to numerical data
    X = vectorize_data(data_df)

    data_df = add_predictions(data_df, model)
    st.write(data_df)

    # Count ratings
    ratings = [0]*5

    # Assuming from 1 to 5
    for rating in data_df['rating']:
        ratings[rating-1] += 1
    
    fig, ax = plt.subplots()
    ax.bar(range(1,6), ratings)
    ax.set_ylabel('Review count')
    ax.set_xlabel('Rating')
    st.pyplot(fig)
    
    st.pyplot(get_common_wordcloud(data_df))

    return

