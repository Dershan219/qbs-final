from tweepy import API, Stream, OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
from datetime import datetime, timedelta
import time
from unidecode import unidecode
import json
import sqlite3
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from nltk.corpus import stopwords
import os
import re
import pickle
from secret import *

os.chdir('C://Users/Andy/Desktop/NTU Courses/Quantitative Business Science/Project/twitter-sentiment')

#%%
# preparing LSTM model-----------------------------------------------------
# custom functions
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def jaccard_coef(y_true, y_pred, smooth=1e-12):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

# loading model and tokenizer
model = load_model(
    'train_model_LSTM.h5',
    custom_objects={
        'jaccard_distance_loss':jaccard_distance_loss,
        'jaccard_coef':jaccard_coef})

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stop_words = set(stopwords.words('english'))
stop_words.update(['rt', 'amp'])
stop_words.remove('not')

def preprocess(text):
    review=re.sub(r"RT |@\S+ |https:\S+|http:\S+|[^a-zA-Z0-9' ]",' ',text)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stop_words]
    review_tokenized=pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=300)
    return review, review_tokenized

# Creating Tweets Database-------------------------------------------------
# update tweets with stream listener
class Listener(StreamListener):
    def __init__(self):
        self.cnt = 0
        self.engine = create_engine('sqlite:///twitter.sqlite')

    def on_data(self, data):
        try:
            self.cnt += 1
            tweet_data = json.loads(data)
            tweet_id = str(tweet_data['id_str'])

            # collecting full tweets
            if 'retweeted_status' in tweet_data:
                if 'extended_tweet' in tweet_data['retweeted_status']:
                    tweet = unidecode(tweet_data['retweeted_status']['extended_tweet']['full_text'])
                else:
                    tweet = unidecode(tweet_data['retweeted_status']['text'])
            else:
                if 'extended_tweet' in tweet_data:
                    tweet = unidecode(tweet_data['extended_tweet']['full_text'])
                else:
                    tweet = unidecode(tweet_data['text'])

            # collecting attributes and sentiment
            time_ts = tweet_data['timestamp_ms']
            time_dt = tweet_data['created_at']
            keywords, x_test = preprocess(tweet)
            sentiment = float(model.predict(x_test, batch_size=1024)[0][0])
            keywords = ', '.join(keywords)
            print("Writing Tweet # {}".format(self.cnt))
            print("Tweet Created at {}".format(time_dt))
            print(tweet)
            print("Tweet Sentiment: {}".format(sentiment))

            # storing tweets into database
            tweets = {
                'id':[tweet_id],
                'time':[time_ts],
                'tweet':[tweet],
                'sentiment':[sentiment],
                'keywords':[keywords]
            }
            df = pd.DataFrame(tweets)
            df.to_sql('tweets', con=self.engine, if_exists='append', index=False)

            # deleting older tweets to limit database size
            with self.engine.connect() as con:
                con.execute(
                    """
                    DELETE FROM tweets
                    WHERE time in(
                    SELECT time
                    FROM(
                    SELECT time,
                    strftime('%M','now') - strftime('%M', datetime(time/1000, 'unixepoch')) as time_passed
                    FROM tweets
                    WHERE time_passed >= 3))
                    """
                )

        except KeyError as e:
            print(str(e))

        return(True)

    def on_error(self, status):
        self.engine.connect().close()
        print(status)

# stream all tweets into database
while True:
    try:
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        twitter_stream = Stream(auth, Listener(), tweet_mode='extended')
        twitter_stream.filter(languages=["en"], track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)