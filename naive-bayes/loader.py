from itertools import chain
from pathlib import Path
import random


def get_tweet_dataset(data_root):
    data_root = Path(data_root)
    samples = dict(
        (
            label.stem,
            tuple(str(sample_path) for sample_path in label.glob("*.json")),
        )
        for label in data_root.iterdir()
        if label.is_dir()
    )

    for label in tuple(samples.keys()):
        if len(samples[label]) == 0:
            del samples[label]

    return samples


def split_train_test(dataset, test_ratio=0.2, shuffle=True):
    labels = tuple(dataset.keys())
    idx_lists = dict((label, list(range(len(dataset[label])))) for label in labels)
    if shuffle:
        for idx_list in idx_lists.values():
            random.shuffle(idx_list)

    train_idx_lists = dict(
        (label, idx_list[round(len(idx_list) * test_ratio) :])
        for label, idx_list in idx_lists.items()
    )
    test_idx_lists = dict(
        (label, idx_list[: round(len(idx_list) * test_ratio)])
        for label, idx_list in idx_lists.items()
    )
    train_set = dict(
        (label, tuple(samples[idx] for idx in train_idx_lists[label]))
        for label, samples in dataset.items()
    )
    test_set = dict(
        (label, tuple(samples[idx] for idx in test_idx_lists[label]))
        for label, samples in dataset.items()
    )

    train_X = tuple(chain(*(train_set[label] for label in labels)))
    test_X = tuple(chain(*(test_set[label] for label in labels)))
    train_y = tuple(chain(*([label] * len(train_set[label]) for label in labels)))
    test_y = tuple(chain(*([label] * len(test_set[label]) for label in labels)))

    return train_X, train_y, test_X, test_y


def split_k_fold(x, y, k=5):
    indices = list(range(len(x)))
    random.shuffle(indices)

    num_samples = len(indices)
    folds = [
        indices[round(num_samples / k * i) : round(num_samples / k * (i + 1))]
        for i in range(k)
    ]

    x_folds = [[x[idx] for idx in fold] for fold in folds]
    y_folds = [[y[idx] for idx in fold] for fold in folds]

    return x_folds, y_folds


def merge_folds(x_folds, y_folds, exclude_idx=-1):
    x, y = [], []
    for i in range(len(x_folds)):
        if i == exclude_idx:
            continue

        x.extend(x_folds[i])
        y.extend(y_folds[i])

    return x, y


def iter_folds(x_folds, y_folds):
    for k in range(len(x_folds)):
        train_x, train_y = merge_folds(x_folds, y_folds, k)
        yield train_x, train_y, x_folds[k], y_folds[k]
