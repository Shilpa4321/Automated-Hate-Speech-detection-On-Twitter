import tweepy
import json
import pandas as pd
#from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib as mpl
import csv
import matplotlib.pyplot as plt

import operator
from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer

#Authentication

consumer_key = 'g7P4mgGL4WTLJISNxh4wpAtYB'
consumer_secret ='E7rxW0FIAi0shd5ylm6cWjGPHwDFbFJRvgvuq08XvLXpp8jWHx'
access_token = '252061060-AKzwKUQfzaiAcPbuMFQtjzCusKlVJ391dFMBrPer'
access_token_secret = 'RB2gyM4DT3CvlUpIB4LSll9xb0ry6d6Wl0G9Q2hN63S31'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) #Interacting with twitter's API
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API (auth) #creating the API object


#Extracting Tweets
results = []
a = input("Enter tweet id: ")
for tweet in tweepy.Cursor (api.search_tweets, q = a , lang = "en").items(100): 
    results.append(tweet)
    
print (type(results))
print (len(results))

#Store tweets data in a dataframe

def tweets_df(results):
    id_list = [tweet.id for tweet  in results]
    data_set = pd.DataFrame(id_list, columns = ["id"])
    
    data_set["text"] = [tweet.text for tweet in results]
    data_set["created_at"] = [tweet.created_at for tweet in results]
    data_set["retweet_count"] = [tweet.retweet_count for tweet in results]
    data_set["user_screen_name"] = [tweet.author.screen_name for tweet in results]
    data_set["user_followers_count"] = [tweet.author.followers_count for tweet in results]
    data_set["user_location"] = [tweet.author.location for tweet in results]
    data_set["Hashtags"] = [tweet.entities.get('hashtags') for tweet in results]
    
    return data_set
    
data_set = tweets_df(results)


# Remove tweets with duplicate text

text = data_set["text"]

for i in range(0,len(text)):
    txt = ' '.join(word for word in text[i] .split() if not word.startswith('https:'))
    #data_set.set_value(i, 'text2', txt)
    data_set.at[i,'text2']=txt
    
    
data_set.drop_duplicates('text2', inplace=True)
data_set.reset_index(drop = True, inplace=True)
data_set.drop('text', axis = 1, inplace = True)
data_set.rename(columns={'text2': 'text'}, inplace=True)

data_set.to_csv('results.csv')
print (data_set)