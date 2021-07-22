from pathlib import Path
from pprint import pformat, pprint

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from loader import get_tweet_dataset, merge_folds, split_k_fold, iter_folds
from tokenizer import create_tokenizer, create_preprocessor
from threading import Thread, Lock
import math
from datetime import datetime as dt
import csv

label_map = {"trump": 0, "biden": 1}
DATASET_DIRECTORY = "../tweet-data/merged_no_dups_no_error"


def candidate_to_binary(candidate):
    return label_map[candidate.lower()]


def train_vectorizer(vectorizer, train_X_filenames):
    vectorizer.fit(train_X_filenames)


def merge_vectorizers(vectorizers, preprocessor, tokenizer):
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
            round(len(dataset) / n_workers * i) : round(
                len(dataset) / n_workers * (i + 1)
            )
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


def train(train_X, train_y, alpha):
    print(f"Splitting data set...")
    start = dt.now()
    train_y = tuple(candidate_to_binary(label) for label in train_y)
    train_X_split = split_dataset_thread(n_workers, train_X)
    train_y_split = split_dataset_thread(n_workers, train_y)
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
    vectorizer = merge_vectorizers(vectorizers, preprocessor, tokenizer)
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    print(f"Fitting classifier...")
    start = dt.now()
    clf = MultinomialNB(alpha=alpha)
    clf_threads = [
        Thread(
            target=train_nb_worker,
            args=(
                clf,
                vectorizer,
                train_X_split[i],
                train_y_split[i],
                (0, 1),
                1000,
            ),
        )
        for i in range(n_workers)
    ]
    for thread in clf_threads:
        thread.start()
    for thread in clf_threads:
        thread.join()
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    return clf, vectorizer


def test(clf, vectorizer, test_X, test_y):
    test_X_split = split_dataset_thread(n_workers, test_X)
    test_y = tuple(candidate_to_binary(label) for label in test_y)
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
    mat = confusion_matrix(y_true=test_y, y_pred=pred_y)
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")
    return mat


if __name__ == "__main__":
    execute_time = dt.now().strftime("%m%d%Y-%H%M%S")
    n_workers = 50

    start = dt.now()
    print(f"Starting at {start.strftime('%m/%d/%Y, %H:%M:%S')}")
    print(f"Loading data set...")
    dataset = get_tweet_dataset(DATASET_DIRECTORY)
    # dataset = dict((label, dataset[label][:1000]) for label in dataset)
    dataset = [
        (sample, label) for label, samples in dataset.items() for sample in samples
    ]
    data_X = [sample[0] for sample in dataset]
    data_y = [sample[1] for sample in dataset]
    print(f"Done! (Running time: {(dt.now() - start).total_seconds()} sec)")

    x_folds, y_folds = split_k_fold(data_X, data_y, k=5)
    test_X, test_y = x_folds[-1], y_folds[-1]
    train_val_X, train_val_y = merge_folds(x_folds[:-1], y_folds[:-1])
    x_folds, y_folds = split_k_fold(train_val_X, train_val_y, k=5)

    accuracies = dict()
    alphas = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    # alphas = [1, 10]
    for alpha in alphas:
        acc = []
        for idx, (train_X, train_y, val_X, val_y) in enumerate(
            iter_folds(x_folds, y_folds)
        ):
            print(f"alpha: {alpha} / K-Fold: {idx}")
            clf, vectorizer = train(train_X, train_y, alpha)
            mat = test(clf, vectorizer, val_X, val_y)
            acc.append(
                (mat[0][0] + mat[1][1])
                / (mat[0][0] + mat[0][1] + mat[1][0] + mat[1][1])
            )
        accuracies[alpha] = sum(acc) / len(acc)

    result_dir = Path("../result_dir/") / execute_time
    result_dir.mkdir(exist_ok=True, parents=True)

    with open(result_dir / "alpha.txt", mode="w") as f:
        f.write(pformat(accuracies))
    pprint(accuracies)

    alpha = max(accuracies, key=lambda k: accuracies[k])
    print(f"final alpha: {alpha}")
    clf, vectorizer = train(train_val_X, train_val_y, alpha)
    mat = test(clf, vectorizer, test_X, test_y)

    csv_filename = result_dir / f"clf.csv"
    txt_filename = result_dir / f"mat.txt"
    print(f"Confusion Matrix:")
    acc = (mat[0][0] + mat[1][1]) / (mat[0][0] + mat[0][1] + mat[1][0] + mat[1][1])
    with open(txt_filename, mode="w") as f:
        print(mat)
        f.write(str(mat))

    print(f"Saving at {csv_filename}...")
    save_feature_csv(
        csv_filename, clf.feature_count_, vectorizer.vocabulary_, label_map
    )

    print("Result accuracy:", acc)
