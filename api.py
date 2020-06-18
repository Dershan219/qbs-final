from tweepy import API, Stream, OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
from datetime import datetime, timedelta
import time
from unidecode import unidecode
import json
from pandas.io.json import json_normalize
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
import re
import pickle

os.chdir('C://Users/Andy/Desktop/NTU Courses/Quantitative Business Science/Project/twitter-sentiment')
consumer_key = "C4dNIgkl43788sKKZ7iNZZy3w"
consumer_secret = "QBfEpERH8d6IdVCPBs4SQaLWfiXgG2b3KRGIV8QrJrBUNJWF2k"
access_token = "712293623665045505-c2NWLT2AQqLVRRS8jcRKxyyHsQ1kpG6"
access_secret = "o82dQvo1DEL2q4Cnp9EehhxsWUW8hs0huV5KwL2tX3dAs"
#%%
# Creating Tweets Database-------------------------------------------------
conn = sqlite3.connect('twitter.db')
c = conn.cursor()

analyzer = SentimentIntensityAnalyzer()

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

model = load_model('train_model_LSTM.h5', custom_objects={'jaccard_distance_loss':jaccard_distance_loss, 'jaccard_coef':jaccard_coef})

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stop_words = set(stopwords.words('english'))
stop_words.remove('not')

def preprocess(text):
    review=re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+',' ',text)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stop_words]
    review_tokenized=pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=300)
    return review, review_tokenized

# create table for tweets
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS tweets (id TEXT PRIMARY KEY, time INTEGER, tweet TEXT, sentiment REAL, keywords TEXT)")
    conn.commit()

create_table()

# update tweets with stream listener
class Listener(StreamListener):
    def on_data(self, data):
        try:
            tweet_data = json.loads(data)
            tweet_id = str(tweet_data['id_str'])
            tweet = unidecode(tweet_data['text'])
            time_ts = tweet_data['timestamp_ms']
            # sentiment = analyzer.polarity_scores(tweet)['compound'] # calculate sentiment scores
            keywords, x_test = preprocess(tweet)
            sentiment = float(model.predict(x_test, batch_size=1024)[0][0])
            keywords = ', '.join(keywords)
            print(sentiment, type(sentiment))
            c.execute("INSERT OR IGNORE INTO tweets (id, time, tweet, sentiment, keywords) VALUES (?, ?, ?, ?, ?)", 
                (tweet_id, time_ts, tweet, sentiment, keywords))
            conn.commit()
        except KeyError as e:
            print(str(e))

        return(True)

    def on_error(self, status):
        print(status)

# stream all tweets into database
while True:
    try:
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        twitter_stream = Stream(auth, Listener())
        twitter_stream.filter(track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)
#%%
# Utility Functions--------------------------------------------------------
# query tweets with keyword for database
def query_table(keyword):
    df = pd.read_sql('SELECT * FROM tweets WHERE tweet LIKE ? ORDER BY time DESC LIMIT 200', conn, params=('%' + keyword + '%', ))
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    return df

# check out top 15 popular tweets
def popular_tweets(keyword, days=7): # yesterday → days = 1
    api = API(auth)
    until = datetime.now().date()-timedelta(days=days)
    results = api.search(keyword, until=until)
    results_df = json_normalize([r._json for r in results])
    tweets_df = results_df.iloc[:,[0,2,3]]
    return tweets_df