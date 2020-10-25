import json
from pathlib import Path
import argparse


def read_json_files(data_dir):
    return sorted(
        [
            file
            for file in Path(data_dir).glob("*.json")
            if file.is_file() and file.stem.isnumeric()
        ],
        key=lambda f: int(f.stem),
    )


def merge_json_files(json_files):
    merged = []
    for file in json_files:
        with open(file, mode="r", encoding="utf8") as data_file:
            json_data = json.load(data_file)
            merged.extend(json_data)
    return merged


def save_json(obj, save_file_name):
    path = Path(save_file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf8") as save_file:
        json.dump(obj, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="../tweet-data/raw",
        help="Directory path containing data.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="../tweet-data/merged",
        help="Directory path where merged data will be saved.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    save = Path(args.save)

    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue

        print(f'Merging candidate "{candidate.stem}"')
        for keyword in candidate.iterdir():
            if not keyword.is_dir():
                continue

            print(f'Merging keyword "{keyword.stem}"')
            merged = merge_json_files(read_json_files(keyword))
            save_json(merged, save / candidate.stem / (keyword.stem + ".json"))

    print("Finished!")
