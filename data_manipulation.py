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


    st.write(data_df)
    st.write(data_df['sentiment'].shape[0])
    ax.hist(data_df['sentiment'], bins=20, range=(0, 2))
    st.pyplot(fig)
    y = pd.get_dummies(data_df['sentiment'])
    
    st.pyplot(get_common_wordcloud(data_df))

    word_count = get_words(data_df)
    fig, ax = plt.subplots()
    ax.hist(word_count.values(), bins=300, range=(0,500), density=True, log=True)
    st.pyplot(fig)

    return
  
def get_dataset_names():
    onlyfiles = [f for f in listdir('datasets') if isfile(join('datasets', f))]
    dataset_names = []
    for file in onlyfiles:
        dataset_names.append(file.removesuffix('.tsv').removesuffix('.csv'))
    return dataset_names

def get_words(data_df):
    common_words=''
    for i in data_df.verified_reviews:
        common_words += ' '.join(str(i).split()) + ' '

    word_count = dict()
    for word in common_words.split():
        if word in word_count.keys():
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

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
