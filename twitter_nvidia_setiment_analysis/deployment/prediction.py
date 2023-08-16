import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load Model

model_lstm = load_model('tweet_sentiment_model.h5')
  
# with open('best_model_logreg.pkl', 'rb') as file_2:
#   logreg_best_model = pickle.load(file_2)


def run():
    # Create Form Input
    with st.form('key=form_sleep_disorder'):
        tweet_content = st.text_input('Tweet Content', placeholder = "masukan tweet disini")

        submitted = st.form_submit_button('Predict')

    data_inf = {
      'tweet_content':tweet_content
    }


    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Predict using ANN
        replace_list = {r"i'm": 'i am',
                r"'re": ' are',
                r"let’s": 'let us',
                r"'s":  ' is',
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan’t": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',
                ',': ' ,',
                '.': ' .',
                '!': ' !',
                '?': ' ?',
                '\s+': ' '}
        def clean_text(text):
            text = text.lower()
            for s in replace_list:
                text = text.replace(s, replace_list[s])
            text = ' '.join(text.split())
            return text
        
        X = data_inf['tweet_content'].apply(lambda p: clean_text(p))

        REPLACE_WITH_SPACE = re.compile("(@)")
        SPACE = " "
        english_stop_words = stopwords.words('english')

        def reviews(reviews):  
            reviews = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in reviews]
            
            return reviews

        def remove_stop_words(corpus):
            removed_stop_words = []
            for review in corpus:
                removed_stop_words.append(
                    ' '.join([word for word in review.split()  if word not in english_stop_words]))
            return removed_stop_words

        def get_stemmed_text(corpus):
            stemmer = PorterStemmer()

            return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

        reviewtweet = reviews(X)
        no_stop_words_tweet = remove_stop_words(reviewtweet)
        stemmed_reviews_tweet = get_stemmed_text(no_stop_words_tweet)

        max_words = 8000


        tokenizer = Tokenizer(
            num_words = max_words,
            filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~'
        )
        tokenizer.fit_on_texts(stemmed_reviews_tweet)
        X = tokenizer.texts_to_sequences(stemmed_reviews_tweet)
        X = pad_sequences(X, maxlen = 200)

        y_pred_inf = model_lstm.predict(X)
        y_pred_inf=np.argmax(y_pred_inf, axis=1)
        y_pred_inf

        st.write('# Predicted Sentiment: ', str(y_pred_inf))
    
if __name__ == '__main__':
    run()