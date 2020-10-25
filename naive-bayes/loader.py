from pathlib import Path
import json
import functools
import random


class TweetDataset:
    def __init__(self, root=None, transforms=None):
        self.root = Path("../tweet-data/split" if root is None else root)
        self.transforms = transforms if transforms is not None else (lambda x: x)
        self.samples = [
            {
                "data": sample,
                "keyword": keyword.stem.lower(),
                "candidate": candidate.stem.lower(),
            }
            for candidate in self.root.iterdir()
            if candidate.is_dir()
            for keyword in candidate.iterdir()
            if keyword.is_dir()
            for sample in keyword.glob("*.json")
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()
        with open(sample["data"], mode="r", encoding="utf8") as file:
            sample["data"] = json.load(file)
        return self.transforms(sample)


class Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class Transforms:
    def __init__(self, transforms=()):
        self.transforms = list(transforms)

    def __call__(self, sample):
        return functools.reduce(lambda acc, el: el(acc), self.transforms, sample)


class TweetDataLoader:
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            yield self.dataset[idx]

    @classmethod
    def create_split_data_loader(
        cls, dataset, ratio, args_for_each_loader=None, kwargs_for_each_loader=None
    ):
        if args_for_each_loader is not None:
            assert len(ratio) == len(args_for_each_loader)
        else:
            args_for_each_loader = [list()] * len(ratio)

        if kwargs_for_each_loader is not None:
            assert len(ratio) == len(kwargs_for_each_loader)
        else:
            kwargs_for_each_loader = [dict()] * len(ratio)

        s = float(sum(ratio))
        ratio = tuple(round(r / s * len(dataset)) for r in ratio)
        indices = [0]
        for i, length in enumerate(ratio):
            indices.append(indices[i] + length)
        indices[-1] = len(dataset)
        start_end_indices = list(zip(indices[:-1], indices[1:]))

        return tuple(
            TweetDataLoader(
                Subset(dataset, range(*start_end_indices[i])),
                *args_for_each_loader[i],
                **kwargs_for_each_loader[i]
            )
            for i in range(len(ratio))
        )
