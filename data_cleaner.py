import nltk
from nltk.corpus import stopwords
from pandas import cut
from textblob import Word
import matplotlib.pyplot as plt
import wordcloud
import re
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# add sentiments based off the numerical rating on a dictionary
# def add_sentiments(data, cutoff=3.0):
#     for review in data:
#         sent = 'Positive' if review['rating'] > cutoff else 'Negative'
#         review['sentiment'] = sent

def apply_sentiments(df, cutoff):
  if df['rating'] >= cutoff:
    return 'Positive'
  elif df['rating'] < cutoff:
    return 'Negative'

def add_sentiments(data_df, cutoff=4.0):
    data_df['sentiment'] = data_df.apply(lambda row: apply_sentiments(row, cutoff), axis=1)
    return data_df

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
def clean_data(data_df):

    stop_words = get_stopwords_from_file()
    debug_output = "original:\n" + data_df['verified_reviews'].head() + '\n'

    # To lowercase
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    debug_output += "lowercase:\n" + data_df['verified_reviews'].head() + '\n'

    # Replacing the special characters and digits/numbers
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(re.sub(r"[^a-zA-Z\s]", "", x) for x in x.split()))
    debug_output += "removed special characters and digits:\n" + data_df['verified_reviews'].head() + '\n'
    
    # Removing stop words
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    debug_output += "removed stopwords:\n" + data_df['verified_reviews'].head() + '\n'

    # Lemmatization
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    debug_output += "lemmatized:\n" + data_df['verified_reviews'].head() + '\n'

    return data_df

def fit_labels(data_df):
    # Encoded the target column
    lb=LabelEncoder()
    data_df['sentiment'] = lb.fit_transform(data_df['sentiment'])

    return data_df

# cleans a single string to be vectorized
def clean_text(text):
    
    stop_words = get_stopwords_from_file()
    # print(f"original: {text}")
    text = text.lower()
    # print(f"lower: {text}")
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # print(f"remove symbols: {text}")
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # print(f"remove stopwords: {text}")
    text = ' '.join([Word(word).lemmatize() for word in text.split()])
    # print(f"lemmatize: {text}")

    return text

def vectorize_text(text, max_words = 500, max_len = 200):
    # Convert reviews to numerical vectors
    tokenizer = Tokenizer(num_words=max_words, split=' ') 
    tokenizer.fit_on_texts([text])
    X = tokenizer.texts_to_sequences([text])
    X = pad_sequences(X, maxlen=max_len)

    return X


def vectorize_data(data_df, max_words = 500, max_len = 200):
    # Convert reviews to numerical vectors
    tokenizer = Tokenizer(num_words=max_words, split=' ') 
    tokenizer.fit_on_texts(data_df['verified_reviews'].values)
    X = tokenizer.texts_to_sequences(data_df['verified_reviews'].values)
    X = pad_sequences(X, maxlen=max_len)
    return X

# returns a plt of the wordcloud of common words
def get_common_wordcloud(data_df):
    common_words=''
    for i in data_df.verified_reviews:
        common_words += ' '.join(str(i).split()) + ' '

    word_cloud = wordcloud.WordCloud().generate(common_words)
    
    fig, ax = plt.subplots()
    ax.imshow(word_cloud, interpolation="bilinear")
    ax.axis("off")

    return fig

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