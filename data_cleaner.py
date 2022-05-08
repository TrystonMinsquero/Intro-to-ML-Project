import nltk
from nltk.corpus import stopwords
from textblob import Word
import matplotlib.pyplot as plt
import wordcloud
import re

# add sentiments based off the numerical rating on a dictionary
def add_sentiments(data, cutoff=3.0):
    for review in data:
        sent = 'Positive' if review['rating'] > cutoff else 'Negative'
        review['sentiment'] = sent

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
    return stopwords.words('english')

# will clean the 'verified_reviews' column to be ready for analysis
def clean(data_df, stop_words):

    # To lowercase
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

    # Replacing the special characters and digits/numbers
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(re.sub(r"[^a-zA-Z]", "", x) for x in x.split()))

    # Removing stop words
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))

    # Lemmatization
    data_df['verified_reviews'] = data_df['verified_reviews'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))

    return data_df

# returns a plt of the wordcloud of common words
def get_common_wordcloud(data_df):
    common_words=''
    for i in data_df.verified_reviews:
        common_words += ' '.join(str(i).split()) + ' '

    word_cloud = wordcloud.WordCloud().generate(common_words)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    return plt