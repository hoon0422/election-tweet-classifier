import nltk
from nltk import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
import re
import json
import unicodedata
from functools import reduce


def create_preprocessor():
    URL_replacement = "[link]"

    def to_json(text: str):
        return json.loads(text)

    def get_full_text(sample):
        if "retweeted_status" in sample:
            sample["full_text"] = sample["retweeted_status"]["full_text"]

        return sample

    def normalize_unicode(sample):
        sample["full_text"] = unicodedata.normalize("NFKD", sample["full_text"])
        return sample

    def to_lower(sample):
        sample["full_text"] = sample["full_text"].lower()
        return sample

    def remove_keyword(sample: dict):
        for keyword in sample["search_keyword"]:
            sample["full_text"] = sample["full_text"].replace(f"#{keyword.lower()}", "")
        return sample

    def replace_link(sample):
        sample["full_text"] = re.sub(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            URL_replacement,
            sample["full_text"],
            flags=re.MULTILINE,
        )
        return sample

    return lambda text: reduce(
        lambda acc, el: el(acc),
        (
            to_json,
            get_full_text,
            normalize_unicode,
            to_lower,
            remove_keyword,
            replace_link,
            lambda sample: sample["full_text"],
        ),
        text,
    )


def create_tokenizer(labels):
    tokenizer = TweetTokenizer()
    lemmatizer = WordNetLemmatizer()
    cached_stopwords = None

    def init():
        nonlocal cached_stopwords
        try:
            wordnet.ensure_loaded()
            cached_stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("punkt")
            nltk.download("stopwords")

    def tokenize(text):
        try:
            return tokenizer.tokenize(text)
        except LookupError:
            return tokenize(text)

    def remove_stopwords(tokenized_text):
        try:
            return [w for w in tokenized_text if w not in cached_stopwords]
        except LookupError:
            return remove_stopwords(tokenized_text)

    def remove_punctuations(tokenized_text):
        try:
            return [w for w in tokenized_text if w not in set(string.punctuation)]
        except LookupError:
            return remove_punctuations(tokenized_text)

    def lemmatize(tokenized_text):
        return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokenized_text]

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

        return tag_dict.get(tag, wordnet.NOUN)

    def remove_words_including_labels(tokenized_text):
        return [
            token
            for token in tokenized_text
            if reduce(lambda acc, el: acc and el not in token, labels, True)
        ]

    init()
    return lambda text: reduce(
        lambda acc, el: el(acc),
        (
            tokenize,
            remove_stopwords,
            remove_punctuations,
            lemmatize,
            remove_words_including_labels,
        ),
        text,
    )
