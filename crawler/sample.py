import json
from pathlib import Path
from typing import List


class Sample:
    path: Path
    keywords: List[str]
    label: str

    @staticmethod
    def get_id_and_sample(path):
        with open(path, encoding="utf8") as f:
            sample = json.load(f)
            keywords = [sample["search_keyword"]]
            label = sample["candidate_label"]
            return sample["id"], Sample(path, keywords, label)

    def __init__(self, path, keywords, label):
        self.path = path
        self.keywords = keywords
        self.label = label


def get_largest_number_in_directory(path: Path):
    if not path.exists():
        return 0

    return (
        max(
            [
                int(sample.stem)
                for sample in path.glob("*.json")
                if sample.stem.isnumeric()
            ]
            + [0]
        )
        + 1
    )
