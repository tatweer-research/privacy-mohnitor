#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd
from datasets import load_dataset

LABELS = [
    "Data Retention",
    "Data Security",
    "Do Not Track",
    "First Party Collection/Use",
    "International and Specific Audiences",
    "Introductory/Generic",
    "Policy Change",
    "Practice not covered",
    "Privacy contact information",
    "Third Party Sharing/Collection",
    "User Access, Edit and Deletion",
    "User Choice/Control",
]


def load_opp_115(directory: str) -> datasets.DatasetDict:
    # define an empty DatasetDict
    combined = datasets.DatasetDict()

    # define available splits
    splits = ["train", "validation", "test"]

    # define label information
    label_info = datasets.Sequence(datasets.ClassLabel(names=LABELS))

    # loop over all splits
    for split in splits:
        # read CSV file corresponding to split
        temp_df = pd.read_csv(
            os.path.join(directory, f"{split}_dataset.csv"),
            header=None,
            names=["text", "label"],
        )

        # aggregate all labels per sentence into a unique list
        temp_df = (
            temp_df.groupby("text")
            .agg(dict(label=lambda x: list(set(x))))
            .reset_index()
        )

        # convert temporary dataframe into HF dataset
        dataset = datasets.Dataset.from_pandas(temp_df, preserve_index=False)

        # convert string labels to integers and store feature information
        dataset = dataset.map(
            lambda examples: {
                "label": [
                    label_info.feature.str2int(labels) for labels in examples["label"]
                ]
            },
            batched=True,
        )
        dataset.features["label"] = label_info

        # insert dataset into combined DatasetDict
        combined[split] = dataset

    return combined


def to_text2text(path='alzoubi36/opp_115'):
    """Convert opp_115 dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path, download_mode='force_redownload')
    label_info = datasets.Sequence(datasets.ClassLabel(names=LABELS))

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        # Add prefix to each datapoint
        dataset = dataset.map(lambda example: {'text': f"opp115 sentence: {example['text']}",
                                               'label': example['label']})

        dataset = dataset.map(
            lambda examples: {
                "label": [
                    '\n'.join(label_info.feature.int2str(labels)) for labels in examples["label"]
                ]
            },
            batched=True,
        )
        dataset_dict[split] = dataset

    return dataset_dict


def label_from_text(label):
    label_info = datasets.Sequence(datasets.ClassLabel(names=LABELS))
    labels = label.split('\n')
    return [label_info.feature.str2int(label_) for label_ in labels]


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\opp_115"
    # dataset_dict = load_opp_115(directory)
    dataset_dict = to_text2text()
    # dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
