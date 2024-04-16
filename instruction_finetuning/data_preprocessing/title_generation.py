import json
import os
from glob import glob
from typing import Any, Dict, List

import datasets
from datasets import load_dataset, Dataset


def parse_multiobject_json_files(paths):
    data = []
    for path in paths:
        file = open(path, encoding="utf-8")
        for line in file:
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                print(f"Skipping line: {line}")
        file.close()
    return data


def create_dataset_from_json(data: list) -> Dataset:
    dataset_dict = {k: [i[k] for i in data] for k in data[0].keys()}
    dataset_dict['id'] = [i for i in range(len(data))]
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def load_title_generation(directory: str) -> datasets.DatasetDict:
    # define DatasetDict for data storage
    combined = datasets.DatasetDict()

    # loop over JSON files
    for json_file in glob(os.path.join(directory, "*.json")):
        # infer split from filename
        filename = os.path.basename(json_file)
        split = (
            "validation"
            if filename.startswith("dev")
            else filename.replace(".json", "")
        )

        data = parse_multiobject_json_files([json_file])
        dataset = create_dataset_from_json(data)
        # convert temp_dict to Dataset and insert into DatasetDict
        combined[split] = dataset

    return combined


def to_text2text(path='alzoubi36/title_generation'):
    """Convert title_generation dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path)
    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        dataset = dataset.map(lambda example: {'text': f"title_generation: {example['text']}",
                                               'label': example['summary']},
                              remove_columns=['summary'])
        dataset_dict[split] = dataset
    return dataset_dict


def flan_text2text(path='alzoubi36/title_generation'):
    """Convert title_generation dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path)
    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        dataset = dataset.map(lambda example: {'text': f"Give a title to the following text: {example['text']}",
                                               'label': example['summary']},
                              remove_columns=['summary'])
        dataset_dict[split] = dataset
    return dataset_dict


if __name__ == "__main__":
    path = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
           r"\title_generation"
    dataset_dict = load_title_generation(path)
    # dataset_dict.push_to_hub('alzoubi36/title_generation')
    print()
