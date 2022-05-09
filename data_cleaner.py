import nltk
from nltk.corpus import stopwords
from textblob import Word
import matplotlib.pyplot as plt
import wordcloud
import re
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# add sentiments based off the numerical rating on a dictionary
def add_sentiments(data, cutoff=3.0):
    for review in data:
        sent = 'Positive' if review['rating'] > cutoff else 'Negative'
        review['sentiment'] = sent

def add_review_length(data_df):
    data_df['review_len'] = data_df['verified_reviews'].astype(str).apply(len)

#checks if there is any null values in the data
def has_null(data_df):
    data_v1 = data_df[['verified_reviews', 'sentiment']]
    for row in data_v1.isnull().sum():
        if row > 0:
            return True
    return False

# needs to be ran once in order to clean data
def download_nltk_requirements():
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')


def populate_stopwords_file():
    file = open('stopwords.json', 'w')
    json.dump(stopwords.words('english'), file)
    file.close()

def get_stopwords_from_file():
    file = open('stopwords.json', 'r')
    stop_words = json.load(file)
    file.close()
    return stop_words
    

# will clean the 'verified_reviews' column to be ready for analysis
def clean(data_df):
    stop_words = get_stopwords_from_file()

    # To lowercase
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

    # Replacing the special characters and digits/numbers
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(re.sub(r"[^a-zA-Z]", "", x) for x in x.split()))

    # Removing stop words
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))

    # Lemmatization
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))

    # Encoded the target column
    lb=LabelEncoder()
    data_df['sentiment'] = lb.fit_transform(data_df['sentiment'])

    return data_df

def vectorize(data_df, num_words=500):
    # Convert reviews to numerical vectors
    tokenizer = Tokenizer(num_words=num_words, split=' ') 
    tokenizer.fit_on_texts(data_df['verified_reviews'].values)
    X = tokenizer.texts_to_sequences(data_df['verified_reviews'].values)
    X = pad_sequences(X)
    return X

# returns a plt of the wordcloud of common words
def get_common_wordcloud(data_df):
    common_words=''
    for i in data_df.verified_reviews:
        common_words += ' '.join(str(i).split()) + ' '

    word_cloud = wordcloud.WordCloud().generate(common_words)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    return plt