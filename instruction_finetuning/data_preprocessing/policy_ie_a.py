#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import cast, Dict

import datasets
from datasets import load_dataset

LABELS = [
    "Other",
    "data-collection-usage",
    "data-security-protection",
    "data-sharing-disclosure",
    "data-storage-retention-deletion",
]


def policy_ie_file_mapping(directory: str, filename: str) -> Dict[str, str]:
    # define patterns for file loading
    files = {}
    files["train"] = os.path.join(directory, "train", filename)
    files["validation"] = os.path.join(directory, "valid", filename)
    files["test"] = os.path.join(directory, "test", filename)
    return files


def load_policy_ie_a(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # load tokens which are common for all sub-tasks
    tokens = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq.in")
    )

    # since this is task A, only load labels
    labels = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "label")
    ).rename_column("text", "label")

    # collect information about label
    label_info = datasets.ClassLabel(names=LABELS)

    # mypy-related specification to sub-type
    tokens = cast(datasets.DatasetDict, tokens)
    labels = cast(datasets.DatasetDict, labels)

    # zip together data
    for split in ["train", "validation", "test"]:
        dataset = datasets.concatenate_datasets([tokens[split], labels[split]], axis=1)
        dataset = dataset.map(
            lambda examples: {
                "label": [label_info.str2int(label) for label in examples["label"]]
            },
            batched=True,
        )
        dataset.features["label"] = label_info
        combined[split] = dataset

    return combined


def to_text2text(path='alzoubi36/policy_ie_a'):
    """Convert privacy_qa dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path)
    # collect information about label
    label_info = datasets.ClassLabel(names=LABELS)

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        # Add prefix to each datapoint
        dataset = dataset.map(lambda example: {'text': f"policy_ie_a sentence: {example['text']}",
                                               'label': example['label']})

        dataset = dataset.map(
            lambda examples: {
                "label": [label_info.int2str(label) for label in examples["label"]]
            },
            batched=True,
        )
        dataset_dict[split] = dataset

    return dataset_dict


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\policy_ie_a"
    # dataset_dict = load_policy_ie_a(directory)
    dataset_dict = to_text2text()
    # dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
