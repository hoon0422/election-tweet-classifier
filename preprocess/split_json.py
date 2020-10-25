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


def get_json_data(json_file):
    with open(json_file, mode="r", encoding="utf8") as data_file:
        return json.load(data_file)


def save_json_data(json_data, file_name):
    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf8") as file:
        json.dump(json_data, file)


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
        default="../tweet-data/split",
        help="Directory path where split data will be saved.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    save = Path(args.save)

    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue

        print(f'Merging candidate "{candidate.stem}"...')
        for keyword in candidate.iterdir():
            if not keyword.is_dir():
                continue

            print(f'Splitting keyword "{keyword.stem}"...')
            file_idx = 0
            for json_file in read_json_files(keyword):
                json_data = get_json_data(json_file)

                if not isinstance(json_data, list):
                    continue

                for sample in json_data:
                    print(f"Saving sample number {file_idx}...")
                    save_json_data(
                        sample,
                        save / candidate.stem / keyword.stem / f"{file_idx}.json",
                    )
                    file_idx += 1

    print("Finished!")
