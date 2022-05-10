import streamlit as st
from data_loader import get_amazon_alexa_data
from data_cleaner import *
import pandas as pd
import json
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from data_uploader import *


def app():

    dataset_name = st.radio("Select Dataset", get_dataset_names())

    data_df = fetch_and_clean_data(dataset_name).copy()


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

def get_dataset_names():
    onlyfiles = [f for f in listdir('datasets') if isfile(join('datasets', f))]
    dataset_names = []
    for file in onlyfiles:
        dataset_names.append(file.removesuffix('.tsv').removesuffix('.csv'))
    return dataset_names

def get_dataset(name):
    onlyfiles = [f for f in listdir('datasets') if isfile(join('datasets', f))]
    for file in onlyfiles:
        if file.removesuffix('.tsv') == name:
            return pd.read_csv(join('datasets', file), sep='\t')
        elif file.removesuffix('.csv') == name:
            return pd.read_csv(join('datasets', file))
    return None

def save_dataset(name, dataset):
    file = open('./datasets/' + name, 'w')
    json.dump(dataset, file)
    file.close()
