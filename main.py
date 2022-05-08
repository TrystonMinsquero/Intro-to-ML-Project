import streamlit as st
from data_loader import get_amazon_alexa_data
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
import redirect as rd

@st.cache(persist=True)
def fetch_and_clean_data():
    print("collected data")
    data = get_amazon_alexa_data()
    # adds sentiment values based on 'verified reviews'
    add_sentiments(data)

    # downloads nltk onto users computer (only need to run once)
    stop_words = download_nltk_requirements()

    # convert data into a pandas dataframe
    data_df = pd.DataFrame(data)
    data_df = clean(data_df, stop_words=stop_words)

    # Encoded the target column
    lb=LabelEncoder()
    data_df['sentiment'] = lb.fit_transform(data_df['sentiment'])

    # Convert reviews to numerical vectors
    tokenizer = Tokenizer(num_words=500, split=' ') 
    tokenizer.fit_on_texts(data_df['verified_reviews'].values)
    X = tokenizer.texts_to_sequences(data_df['verified_reviews'].values)
    X = pad_sequences(X)

    return data_df, X

data_df, X = fetch_and_clean_data()

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

model = create_model(X)


# Create LTSM Model 
# Get inputs
helpLink = "https://keras.io/getting_started/faq/#what-do-sample-batch-and-epoch-mean"

form = st.form(key="Test_parameters")
submit = form.form_submit_button("Start Training")
form.title("Training Parameters")

test_size_percent = form.number_input("Test Size Percentage", 0.0, 1.0,
    help="default is .3")

batch_size = form.number_input("Batch Size", 1, len(data_df),
    help=f"See {helpLink} for more info about the variables")

epochs = form.number_input("Epochs", 1, 
    help=f"See {helpLink} for more info about the variables")


if submit:
    #Splitting the data into training and testing
    y=pd.get_dummies(data_df['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size_percent, random_state = 42)
    with rd.stdout:
        model.fit(X_train, y_train, epochs = epochs, batch_size=batch_size, verbose = 'auto')
        loss, accuracy = model.evaluate(X_test, y_test)
        st.write(f"Loss: {round(loss,4)}   and   Accuracy: {round(accuracy, 4)}")


st.pyplot(get_common_wordcloud(data_df))

# st.write(data_df)