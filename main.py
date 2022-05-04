from asyncio.windows_events import NULL
import streamlit as st
from data_loader import data
from data_cleaner import *
import pandas as pd
# Importing required libraries
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 

# adds sentiment values based on 'verified reviews'
add_sentiments(data)
download_nltk_requirements()

# convert data into a pandas dataframe
data_df = pd.DataFrame(data)

data_df = clean(data_df)


st.pyplot(get_common_wordcloud(data_df))

st.write(data_df)