#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import cast, Dict

import datasets

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


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\policy_ie_a"
    dataset_dict = load_policy_ie_a(directory)
    dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
