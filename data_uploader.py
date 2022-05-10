import streamlit as st
from data_cleaner import *
import pandas as pd
from os import listdir
from os.path import isfile, join


def app():

    st.title("Dataset Uploader")

    with st.form("Import Training Dataset"):
        name = st.text_input("Name of dataset", value="dataset")
        text_column_name = st.text_input("Text column name", value="verified_reviews")
        rating_column_name = st.text_input("Rating column name", value='rating')
        file = st.file_uploader("Dataset file", type=['csv', 'tsv'])
        data = None
        if file:
            print(file)
            if 'csv' in file.name:
                data = pd.read_csv(file)
                st.write(data)
            elif 'tsv' in file.name:
                data = pd.read_csv(file, sep='\t')
                st.write(data)

        submit = st.form_submit_button("Save file")

        if submit and data is not None:
            data.rename({text_column_name: 'verified_reviews', rating_column_name: 'rating'})
            if 'csv' in file.name:
                data.to_csv(join('datasets', name + '.csv'), columns=[text_column_name, rating_column_name])
                st.success('Saved file to ' + join('datasets', name + '.csv'))
            elif 'tsv' in file.name:
                data.to_csv(join('datasets', name + '.tsv'), sep='\t', columns=[text_column_name, rating_column_name])
                st.success('Saved file to ' + join('datasets', name + '.tsv'))
        elif submit:
            st.warning("Need to upload a file to save it")


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

@st.cache(persist=True, allow_output_mutation=True)
def fetch_and_clean_data(name):
    
    data_df = get_dataset(name)
    # adds sentiment values based on 'verified reviews'
    add_sentiments(data_df)

    # downloads nltk onto users computer (only need to run once)
    download_nltk_requirements()

    data_df = clean_data(data_df)
    data_df = fit_labels(data_df)

    return data_df