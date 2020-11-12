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
