from pprint import pprint

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from loader import get_tweet_dataset, split_train_test
from tokenizer import create_tokenizer, create_preprocessor
from threading import Thread, Lock
import math
from datetime import datetime as dt
import csv

label_map = {"trump": 0, "biden": 1}


def candidate_to_binary(candidate):
    return label_map[candidate.lower()]


def train_vectorizer(vectorizer, train_X_filenames):
    print(f"Fitting with #{len(train_X_filenames)} files...")
    vectorizer.fit(train_X_filenames)


def merge_vectorizers(vectorizers):
    merged_voca = set(
        word for vectorizer in vectorizers for word in vectorizer.vocabulary_.keys()
    )
    v = CountVectorizer(
        input="filename",
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        vocabulary=dict(
            (word, idx) for word, idx in zip(merged_voca, range(len(merged_voca)))
        ),
    )
    v._validate_vocabulary()
    return v


def split_dataset_thread(n_workers, dataset):
    return [
        dataset[
            round(len(dataset) / n_workers)
            * i : round(len(dataset) / n_workers)
            * (i + 1)
        ]
        for i in range(n_workers)
    ]


lock = Lock()


def train_nb_worker(
    nb, vectorizer, train_sample_x, train_sample_y, classes, batch_size=-1
):
    if batch_size == -1:
        batch_size = len(train_sample_x)

    for i in range(math.ceil(len(train_sample_x) / batch_size)):
        x = vectorizer.transform(
            train_sample_x[i * batch_size : (i + 1) * batch_size]
        ).toarray()
        y = train_sample_y[i * batch_size : (i + 1) * batch_size]
        with lock:
            nb.partial_fit(x, y, classes=classes)


def predict_nb_worker(nb, vectorizer, sample_x, prediction, batch_size=-1):
    if batch_size == -1:
        batch_size = len(sample_x)

    for i in range(math.ceil(len(sample_x) / batch_size)):
        x = vectorizer.transform(
            sample_x[i * batch_size : (i + 1) * batch_size]
        ).toarray()
        pred = nb.predict(x)
        for j in range(len(pred)):
            prediction[i * batch_size + j] = pred[j]


def save_feature_csv(filename, feature_count, vocabulary, label_map):
    feature_word_map = dict((v, k) for k, v in vocabulary.items())
    fields = list(label_map.keys())
    with open(filename, mode="w", encoding="utf8") as file:
        writer = csv.DictWriter(file, fieldnames=["word"] + fields)
        writer.writeheader()
        for feature_num, feature in feature_word_map.items():
            row = {"word": feature}
            row.update(
                dict(
                    (field, feature_count[label_map[field]][feature_num])
                    for field in fields
                )
            )
            writer.writerow(row)


if __name__ == "__main__":
    filename = "clf.csv"
    n_workers = 50
    start = dt.now()
    print(f"Starting at {start.strftime('%m/%d/%Y, %H:%M:%S')}")
    print(f"Loading data set...")
    dataset = get_tweet_dataset("../tweet-data/merged_no_dups")
    # dataset = dict((label, dataset[label][:100]) for label in dataset)
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    print(f"Splitting data set...")
    start = dt.now()
    train_X, train_y, test_X, test_y = split_train_test(
        dataset, test_ratio=0.2, shuffle=True
    )
    train_y = tuple(candidate_to_binary(label) for label in train_y)
    test_y = tuple(candidate_to_binary(label) for label in test_y)
    train_X_split = split_dataset_thread(n_workers, train_X)
    test_X_split = split_dataset_thread(n_workers, test_X)
    train_y_split = split_dataset_thread(n_workers, train_y)
    test_y_split = split_dataset_thread(n_workers, test_y)
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    preprocessor = create_preprocessor()
    tokenizer = create_tokenizer(["trump", "biden"])

    print(f"Fitting Vectorizer with training set (length: {len(train_X)})...")
    start = dt.now()
    vectorizers = [
        CountVectorizer(
            input="filename", tokenizer=tokenizer, preprocessor=preprocessor
        )
        for _ in range(n_workers)
    ]
    vectorizer_threads = [
        Thread(target=train_vectorizer, args=(vectorizers[i], train_X_split[i]))
        for i in range(n_workers)
    ]
    for thread in vectorizer_threads:
        thread.start()
    for thread in vectorizer_threads:
        thread.join()
    vectorizer = merge_vectorizers(vectorizers)
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    print(f"Fitting classifier...")
    start = dt.now()
    clf = MultinomialNB()
    clf_threads = [
        Thread(
            target=train_nb_worker,
            args=(clf, vectorizer, train_X_split[i], train_y_split[i], (0, 1), 1000),
        )
        for i in range(n_workers)
    ]
    for thread in clf_threads:
        thread.start()
    for thread in clf_threads:
        thread.join()
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    print("Testing classifier...")
    start = dt.now()
    pred_y_split = [[0] * len(test_X_split[i]) for i in range(n_workers)]
    pred_threads = [
        Thread(
            target=predict_nb_worker,
            args=(clf, vectorizer, test_X_split[i], pred_y_split[i], 100),
        )
        for i in range(n_workers)
    ]
    for thread in pred_threads:
        thread.start()
    for thread in pred_threads:
        thread.join()

    pred_y = []
    for pys in pred_y_split:
        pred_y.extend(pys)
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true=test_y, y_pred=pred_y))

    print(f"Saving at {filename}...")
    save_feature_csv(filename, clf.feature_count_, vectorizer.vocabulary_, label_map)
