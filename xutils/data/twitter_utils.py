"""Twitter Utils"""

import tweepy
from textblob import TextBlob
import numpy as np


def get_api(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)


def search(query=None, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None, api=None):
    if api is None:
        api = get_api(consumer_key, consumer_secret, access_token, access_token_secret)
    return api.search(query)


def get_sentiment(query=None, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None,
                  api=None,
                  search_results=None,
                  method=np.mean, threshold=0.0, attribute='polarity'):

    if search_results is None:
        search_results = search(query, consumer_key, consumer_secret, access_token, access_token_secret, api)

    sentiments = []
    for result in search_results:
        result_text = result.text
        blob = TextBlob(result_text)
        value = getattr(blob.sentiment, attribute)
        if abs(value) > threshold:
            sentiments.append(value)

    return method(sentiments)

