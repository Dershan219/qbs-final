import tweepy
import pandas as pd
from pandas.io.json import json_normalize
import sqlite3
from secret import *

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
results = api.search('Trump')

results_df = json_normalize([r._json for r in results])
tweets_df = results_df.iloc[:,[0,2,3]]
