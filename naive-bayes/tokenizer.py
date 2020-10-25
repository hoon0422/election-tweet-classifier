import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
from loader import Transforms
import re

lemmatizer = WordNetLemmatizer()


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


def get_full_text(sample):
    if "retweeted_status" in sample["data"]:
        sample["data"] = sample["data"]["retweeted_status"]
    sample["data"] = sample["data"]["full_text"]

    return sample


def to_lower(sample):
    sample["data"] = sample["data"].lower()
    return sample


def remove_keyword(sample):
    sample["data"] = sample["data"].replace(f'#{sample["keyword"].lower()}', "")
    return sample


def replace_link(sample):
    replaced_text = "[link]"
    sample["data"] = re.sub(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        replaced_text,
        sample["data"],
        flags=re.MULTILINE,
    )
    return sample


def tokenize(sample):
    try:
        sample["data"] = word_tokenize(sample["data"])
        return sample
    except LookupError:
        nltk.download("punkt")
        return tokenize(sample)


def remove_stopwords(sample):
    try:
        sample["data"] = [
            w for w in sample["data"] if w not in set(stopwords.words("english"))
        ]
        return sample
    except LookupError:
        nltk.download("stopwords")
        return remove_stopwords(sample)


def remove_punctuations(sample):
    try:
        sample["data"] = [w for w in sample["data"] if w not in set(string.punctuation)]
        return sample
    except LookupError:
        nltk.download("punkt")
        return remove_punctuations(sample)


def lemmatize(sample):
    sample["data"] = [
        lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in sample["data"]
    ]
    return sample


transforms = Transforms(
    [
        # sample['data] is `str` type
        get_full_text,
        to_lower,
        remove_keyword,
        replace_link,
        tokenize,
        # Now, sample['data] is `List[str]` type
        remove_stopwords,
        remove_punctuations,
        lemmatize,
    ]
)
