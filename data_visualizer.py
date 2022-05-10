import streamlit as st
from data_cleaner import *
import pandas as pd
import matplotlib.pyplot as plt
from data_uploader import *


def app():

    dataset_name = st.radio("Select Dataset", get_dataset_names())

    data_df = fetch_and_clean_data(dataset_name)


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

