from datetime import datetime
from typing import Union

import tweepy
import simplejson as json
from pathlib import Path
from dotenv import load_dotenv
import os


class Searcher:
    keyword: str
    label: str
    save_directory: Union[Path, str]
    count: int

    def __init__(self, keyword, label, root_directory, api, count=100, **cursor_kwargs):
        self.keyword = keyword
        self.label = label
        self.save_directory = Path(root_directory) / label / keyword[1:]
        self.count = count
        self.cursor = tweepy.Cursor(
            api.search,
            q=self.keyword,
            count=count,
            tweet_mode="extended",
            **cursor_kwargs,
        )

    def request_data(self):
        items = list(self.cursor.items(self.count))
        return [status._json for status in items]

    def save_json_data(self, json_list):
        self.save_directory.mkdir(parents=True, exist_ok=True)
        starting_number = self._get_largest_number_in_directory()
        for json_data in json_list:
            json_data["search_keyword"] = self.keyword
            json_data["candidate_label"] = self.label

            with open(
                Path(self.save_directory, f"{starting_number}.json"),
                mode="w",
                encoding="utf8",
            ) as json_file:
                json_file.write(json.dumps(json_data))

            starting_number += 1

    def _get_largest_number_in_directory(self):
        if not self.save_directory.exists():
            return 0

        return (
            max(
                [
                    int(sample.stem)
                    for sample in self.save_directory.glob("*.json")
                    if sample.stem.isnumeric()
                ]
                + [0]
            )
            + 1
        )


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
        auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, timeout=16 * 60
    )
    print("Accessed Twitter API!")

    root_directory = Path(
        "../tweet-data/raw/" + datetime.now().strftime("%m%d%Y-%H%M%S")
    )
    trump_label = "trump"
    trump_keywords = [
        "#TrumpPence",
        "#trump2020",
        "#TrumpPence2020",
        "#4MoreYears",
        "#TRUMP2020ToSaveAmerica",
        "#TrumpTrain",
        "#Trump2020LandslideVictory",
    ]
    biden_label = "biden"
    biden_keywords = [
        "#BidenHarris",
        "#biden2020",
        "#BidenHarrisToSaveAmerica",
        "#BidenHarris2020",
        "#BlueWave2020",
        "#VoteThemOut",
    ]

    print("Creating cursors...")
    count = 100
    trump_searchers = [
        Searcher(keyword, trump_label, root_directory, api, count=count, lang="en")
        for keyword in trump_keywords
    ]
    biden_searchers = [
        Searcher(keyword, biden_label, root_directory, api, count=count, lang="en")
        for keyword in biden_keywords
    ]

    searchers = []
    min_length = min(len(trump_searchers), len(biden_searchers))
    for i in range(min_length):
        searchers.append(trump_searchers[i])
        searchers.append(biden_searchers[i])
    searchers.extend(trump_searchers[min_length:])
    searchers.extend(biden_searchers[min_length:])
    print("Cursor created!")

    num_of_pages = 1000
    done_searchers = set()
    for i in range(num_of_pages):
        validate_searchers = (
            searcher for searcher in searchers if searcher not in done_searchers
        )
        for searcher in validate_searchers:
            json_list = searcher.request_data()
            if len(json_list) == 0:
                done_searchers.add(searcher)
                continue
            searcher.save_json_data(json_list)
            print(f'Request #{i} for keyword "{searcher.keyword}" has been saved.')

        if len(done_searchers) == len(searchers):
            break
