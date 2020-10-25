from pprint import pprint

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from loader import TweetDataLoader, TweetDataset
from tokenizer import transforms


def candidate_to_binary(candidate):
    if candidate == "trump":
        return 0
    else:
        return 1


print(f"Loading data set...")
dataset = TweetDataset("../tweet-data/split", transforms=transforms)
data_loader = TweetDataLoader(dataset, shuffle=True)
data = [
    (sample["data"], candidate_to_binary(sample["candidate"])) for sample in data_loader
]
data_X, data_y = [sample[0] for sample in data], [sample[1] for sample in data]
train_X, test_X, train_y, test_y = train_test_split(
    data_X, data_y, train_size=0.8, random_state=0
)

vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
print(f"Vectorizing training set (length: {len(train_X)})...")
train_X = vectorizer.fit_transform(train_X).toarray()
print(f"Vectorizing test set (length: {len(test_X)})...")
test_X = vectorizer.transform(test_X).toarray()

print("Fitting classifier...")
clf = MultinomialNB()
clf.fit(train_X, train_y)

if __name__ == "__main__":
    print("Testing classifier...")
    pred_y = clf.predict(test_X)
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true=test_y, y_pred=pred_y))
