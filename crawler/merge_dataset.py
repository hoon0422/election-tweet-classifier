from sample import get_largest_number_in_directory
from pathlib import Path
from shutil import copy2, copytree

if __name__ == "__main__":
    root_directory = Path("../tweet-data")
    save_dir = root_directory / "merged2"
    datasets = (
        "11022020-134320_no_dups",
        "merged",
    )
    datasets = tuple(root_directory / dataset for dataset in datasets)

    for dataset in datasets:
        if not dataset.exists():
            continue

            # for candidate_dir in dataset.iterdir():
            #     if not candidate_dir.is_dir():
            #         continue

        for keyword_dir in candidate_dir.iterdir():
            if not keyword_dir.is_dir():
                continue

            existing_keyword_dir = save_dir / candidate_dir.stem / keyword_dir.stem
            if not existing_keyword_dir.exists():
                copytree(keyword_dir, existing_keyword_dir)
                continue

            ln = get_largest_number_in_directory(existing_keyword_dir)
            for json_file in keyword_dir.glob("*.json"):
                copy2(
                    json_file,
                    existing_keyword_dir / (str(int(json_file.stem) + ln) + ".json"),
                )
