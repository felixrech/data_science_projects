import pandas as pd
import itertools as it
from joblib import Memory
from time import sleep, time
from snscrape.modules.twitter import TwitterSearchScraper, Tweet


cache = Memory(".cache")


@cache.cache
def _get_tweets(search_term) -> list[Tweet]:
    return list(TwitterSearchScraper(search_term).get_items())


def get_tweets(search_term, min_faves=50, since="2023-01-01", until="2023-05-01"):
    tweets = []
    for since, until in it.pairwise(pd.date_range(since, until, freq="D")):
        start = time()
        tweets += _get_tweets(
            f"{search_term} min_faves:{min_faves} lang:en "
            f"since:{since.date()} until:{until.date()}"
        )
        # If we did not use cached data, sleep to avoid rate limits
        if time() - start > 1:
            sleep(10)
    return tweets


def _sentiment_string(text):
    # TODO: changeable aspect
    text = text.replace("\n", " ")
    return f"[CLS]{text} [SEP]Putin[SEP]"


def get_tweets_df(search_term, min_faves=50, since="2023-01-01", until="2023-05-01"):
    tweets = get_tweets(search_term, min_faves, since, until)
    df = pd.DataFrame(
        dict(
            id=[tweet.id for tweet in tweets],
            date=[tweet.date for tweet in tweets],
            text=[tweet.renderedContent for tweet in tweets],
            username=[tweet.user.username for tweet in tweets],
            coordinates=[tweet.coordinates for tweet in tweets],
            user_location=[tweet.user.location for tweet in tweets],
        )
    )
    return df.assign(sentiment_string=lambda df: df["text"].map(_sentiment_string))
