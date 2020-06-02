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
from secret import *

# check out top 15 popular tweets
def popular_tweets(keyword, days=7): # yesterday → days = 1
    api = API(auth)
    until = datetime.now().date()-timedelta(days=days)
    results = api.search(keyword, result_type='popular', until=until)
    results_df = json_normalize([r._json for r in results])
    tweets_df = results_df.iloc[:,[0,2,3]]
    return tweets_df

#%%
conn = sqlite3.connect('twitter.db')
c = conn.cursor()

analyzer = SentimentIntensityAnalyzer()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS tweets (id TEXT PRIMARY KEY, time INTEGER, tweet TEXT, sentiment REAL)")
    conn.commit()

create_table()

class Listener(StreamListener):
    def on_data(self, data):
        try:
            data = json.loads(data)
            tweet_id = str(data['id_str'])
            tweet = unidecode(data['text'])
            time_ts = data['timestamp_ms']
            sentiment = analyzer.polarity_scores(tweet)['compound']
            print(time_ts, tweet_id, tweet, sentiment)
            # sentiment = model.predict(tweet)
            c.execute("INSERT OR IGNORE INTO tweets (id, time, tweet, sentiment) VALUES (?, ?, ?, ?)", 
                (tweet_id, time_ts, tweet, sentiment))
            conn.commit()
        except KeyError as e:
            print(str(e))

        return(True)

    def on_error(self, status):
        print(status)

#%%
while True:
    try:
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        twitter_stream = Stream(auth, Listener())
        twitter_stream.filter(track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)

