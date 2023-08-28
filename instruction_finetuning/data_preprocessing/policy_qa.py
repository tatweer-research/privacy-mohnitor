#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from glob import glob
from typing import Any, Dict, List

import datasets
from datasets import load_dataset


def load_policy_qa(directory: str) -> datasets.DatasetDict:
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

        # define temporarily dictionary
        temp_dict: Dict[str, List[Any]] = {
            "id": [],
            "title": [],
            "context": [],
            "question": [],
            "answers": [],
        }

        # read JSON file
        with open(json_file, "r") as input_file_stream:
            dataset = json.load(input_file_stream)

        # loop over data and save to dictionray
        for article in dataset["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    temp_dict["id"].append(qa["id"])
                    temp_dict["title"].append(title)
                    temp_dict["context"].append(context)
                    temp_dict["question"].append(qa["question"])
                    temp_dict["answers"].append(
                        {
                            "text": [answer["text"] for answer in qa["answers"]],
                            "answer_start": [
                                answer["answer_start"] for answer in qa["answers"]
                            ],
                        }
                    )

        # convert temp_dict to Dataset and insert into DatasetDict
        combined[split] = datasets.Dataset.from_dict(temp_dict)

    return combined


def to_text2text(path='alzoubi36/policy_qa'):
    """Converts the policy_qa dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path)

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]

        dataset = dataset.map(lambda example: {'text': f"policy_qa question: {example['question']} "
                                                       f"context: {example['context']}",
                                               'label': f"{example['answers']['text'][0]}"},
                              remove_columns=['question', 'context', 'answers', 'id', 'title'])
        dataset_dict[split] = dataset

    return dataset_dict


def label_from_text(label):
    raise NotImplementedError("No need to implement this function for policy_qa dataset")


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\policy_qa"
    # dataset_dict = load_policy_qa(directory)
    dataset_dict = to_text2text()
    # dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
