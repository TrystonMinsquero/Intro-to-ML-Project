import streamlit as st
from data_cleaner import *
import pandas as pd
import matplotlib.pyplot as plt
from data_uploader import *
from model_design import get_model_names, get_model
from predict import add_predictions



def app():
    st.title("Data Visualization")

    st.header("Input Data")

    dataset_name = st.radio("Select Dataset", get_dataset_names())
    
    # get raw data
    data_df = get_dataset(dataset_name)

    st.subheader("Raw data for " + dataset_name)
    st.write(data_df)

    # clean data 
    data_df = clean_data(data_df)
    # add sentiments based off rating
    data_df = add_sentiments(data_df)
    # fit labels to cleaned data
    data_df = fit_labels(data_df)

    st.subheader("Cleaned data for " + dataset_name)
    st.write(data_df)
    
    st.subheader("Word Cloud for the cleaned data for " + dataset_name)
    st.pyplot(get_common_wordcloud(data_df))

    st.header("Output Data")

    modelname = st.radio('Choose Model to Predict with', get_model_names())

    model = get_model(modelname)

    # convert text to numerical data
    X = vectorize_data(data_df)

    data_df = add_predictions(data_df, model)
    st.write(data_df)

    # Count ratings
    ratings = [0]*5

    # Assuming from 1 to 5
    for rating in data_df['rating']:
        ratings[rating-1] += 1

    # Categorize the sentiments by rating
    def sentiment_group(g):
        return rating_groups.get_group(g)['predicted_sentiment'].to_list()

    rating_groups = data_df.groupby('rating')
    rating_sentiments = [sentiment_group(g) for g in rating_groups.groups]
    
    fig, ax = plt.subplots()
    ax.bar(range(1,6), ratings)
    ax.set_ylabel('Review count')
    ax.set_xlabel('Rating')

    st.subheader("Frequency of ratings")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.boxplot(rating_sentiments)
    ax.set_ylabel('Sentiments')
    ax.set_xlabel('Rating')
    
    st.subheader("Rating by sentiment") # Please title this better
    st.pyplot(fig)

    return

