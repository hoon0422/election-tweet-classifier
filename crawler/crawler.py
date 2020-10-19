import time

import tweepy
import json
from pathlib import Path
from dotenv import load_dotenv
import os


def return_items(cursor):
    return cursor.items()


if __name__ == "__main__":
    print("Accessing Twitter API...")
    load_dotenv(dotenv_path=Path("../.env"))
    consumer_key = os.getenv("CONSUMER_KEY")
    consumer_secret = os.getenv("CONSUMER_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(
        auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, timeout=15 * 60
    )
    print("Accessed Twitter API!")

    trump_keyword = "#trump2020"
    biden_keyword = "#biden2020"

    trump_dir = Path(f"../tweet-data/raw/trump/{trump_keyword[1:]}")
    biden_dir = Path(f"../tweet-data/raw/biden/{biden_keyword[1:]}")
    trump_dir.mkdir(parents=True, exist_ok=True)
    biden_dir.mkdir(parents=True, exist_ok=True)

    print("Creating cursors...")
    count = 60
    trump_cursor = tweepy.Cursor(
        api.search, q=trump_keyword, count=count, tweet_mode="extended"
    )
    biden_cursor = tweepy.Cursor(
        api.search, q=biden_keyword, count=count, tweet_mode="extended"
    )
    print("Cursor created!")

    for i in range(167):
        trump_items = list(trump_cursor.items(count))
        trump_json_list = [status._json for status in trump_items]
        with open(Path(trump_dir, f"{i}.json"), "w") as file:
            file.write(json.dumps(trump_json_list))
        print(f'Request #{i} for keyword "{trump_keyword}" has been saved.')

        biden_items = list(biden_cursor.items(count))
        biden_json_list = [status._json for status in biden_items]
        with open(Path(biden_dir, f"{i}.json"), "w") as file:
            file.write(json.dumps(biden_json_list))
        print(f'Request #{i} for keyword "{biden_keyword}" has been saved.')
