import streamlit as st
from data_loader import get_amazon_alexa_data
from data_cleaner import *
import pandas as pd
import json
import matplotlib.pyplot as plt


def app():
    data_df, X = fetch_and_clean_data()
    
    # add_review_length(data_df)
    st.write(data_df)
    st.write(data_df['sentiment'].shape[0])
    plt.hist(data_df['sentiment'], bins=20, range=(0, 2))
    # st.pyplot(plt.show())
    y = pd.get_dummies(data_df['sentiment'])
    
    st.pyplot(get_common_wordcloud(data_df))
    
    # plt.scatter(X[0], y)
    # st.pyplot(plt.show())

    return

def get_dataset(name):
    file = open('./datasets/' + name, 'r')
    dataset = json.load(file)
    file.close()
    return dataset

def save_dataset(name, dataset):
    file = open('./datasets/' + name, 'w')
    json.dump(dataset, file)
    file.close()

@st.cache(persist=True)
def fetch_and_clean_data():
    
    data = get_amazon_alexa_data()
    # adds sentiment values based on 'verified reviews'
    add_sentiments(data)

    # downloads nltk onto users computer (only need to run once)
    download_nltk_requirements()

    # convert data into a pandas dataframe
    data_df = pd.DataFrame(data)
    data_df = clean(data_df)
    X = vectorize(data_df)

    return data_df, X