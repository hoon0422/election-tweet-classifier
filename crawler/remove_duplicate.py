from pathlib import Path
import json
from typing import Dict
from sample import Sample

if __name__ == "__main__":
    root = Path("../tweet-data/merged")
    biden_dir = root / "biden"
    trump_dir = root / "trump"
    save_dir = root.parent / (root.stem + "_no_dups")

    # remove samples duplicated within the same keywords
    biden_kw_data_lists = [
        list(Sample.get_id_and_sample(file_path) for file_path in path.glob("*.json"))
        for path in biden_dir.glob("*")
        if path.is_dir()
    ]
    trump_kw_data_lists = [
        list(Sample.get_id_and_sample(file_path) for file_path in path.glob("*.json"))
        for path in trump_dir.glob("*")
        if path.is_dir()
    ]

    biden_kw_data_dicts = [dict(keyword_data) for keyword_data in biden_kw_data_lists]
    trump_kw_data_dicts = [dict(keyword_data) for keyword_data in trump_kw_data_lists]
    num_of_kw_duplicated = sum(
        len(biden_kw_data_lists[idx]) - len(biden_kw_data_dicts[idx])
        for idx in range(len(biden_kw_data_dicts))
    ) + sum(
        len(trump_kw_data_lists[idx]) - len(trump_kw_data_dicts[idx])
        for idx in range(len(biden_kw_data_dicts))
    )
    print(f"{num_of_kw_duplicated} keyword duplicated")

    # find samples duplicated within the same label and add keywords to the samples
    biden_data: Dict[str, Sample] = dict()
    trump_data: Dict[str, Sample] = dict()
    keyword_data_lists = [
        (biden_data, biden_kw_data_dicts),
        (trump_data, trump_kw_data_dicts),
    ]
    label_duplicated_num = 0
    for data, keyword_data_list in keyword_data_lists:
        for keyword_data in keyword_data_list:
            for sample_id, sample in keyword_data.items():
                if sample_id in data:
                    data[sample_id].keywords.extend(sample.keywords)
                    label_duplicated_num += 1
                else:
                    data[sample_id] = sample
    print(f"{label_duplicated_num} label duplicated")

    # remove samples duplicated over labels
    duplicated_over_labels = set(
        sample_id for sample_id in biden_data if id in trump_data
    )
    for sample_id in duplicated_over_labels:
        del biden_data[sample_id]
        del trump_data[sample_id]
    print(f"{len(duplicated_over_labels)} over-label duplicated")

    # save non duplicates
    save_dir_biden = save_dir / "biden"
    save_dir_trump = save_dir / "trump"
    save_dir_biden.mkdir(parents=True, exist_ok=True)
    save_dir_trump.mkdir(parents=True, exist_ok=True)
    for save_dir_candidate, data_candidate in [
        (save_dir_biden, biden_data),
        (save_dir_trump, trump_data),
    ]:
        for idx, sample in enumerate(data_candidate.values(), 1):
            with open(sample.path, encoding="utf8") as f:
                sample_data = json.load(f)
                sample_data["search_keyword"] = sample.keywords

            with open(
                save_dir_candidate / f"{idx}.json", mode="w", encoding="utf8"
            ) as f:
                f.write(json.dumps(sample_data))
