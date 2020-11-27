from pathlib import Path
import json
from sample import Sample

filtered_count = 0


def filter_tweet(sample_and_text):
    global filtered_count
    sample, text = sample_and_text
    text = text.lower()
    filtered_text = text
    for keyword in sample.keywords:
        filtered_text = filtered_text.replace(f"{keyword.lower()}", "")

    if text == filtered_text:
        filtered_count += 1
        return False
    return True


if __name__ == "__main__":
    root = Path("../tweet-data/merged_no_dups")
    biden_dir = root / "biden"
    trump_dir = root / "trump"
    save_dir = root.parent / (root.stem + "_no_error")

    biden_tweets = filter(
        filter_tweet,
        [
            Sample.get_sample_and_text(file_path)
            for file_path in biden_dir.glob("*.json")
        ],
    )

    trump_tweets = filter(
        filter_tweet,
        [
            Sample.get_sample_and_text(file_path)
            for file_path in trump_dir.glob("*.json")
        ],
    )

    # save tweets without errors
    save_dir_biden = save_dir / "biden"
    save_dir_trump = save_dir / "trump"
    save_dir_biden.mkdir(parents=True, exist_ok=True)
    save_dir_trump.mkdir(parents=True, exist_ok=True)
    for save_dir_candidate, tweets in [
        (save_dir_biden, biden_tweets),
        (save_dir_trump, trump_tweets),
    ]:
        for idx, sample_and_text in enumerate(tweets, 1):
            sample = sample_and_text[0]
            with open(sample.path, encoding="utf8") as f:
                sample_data = json.load(f)

            with open(
                save_dir_candidate / f"{idx}.json", mode="w", encoding="utf8"
            ) as f:
                f.write(json.dumps(sample_data))

    print(f"filtered count: {filtered_count}")
