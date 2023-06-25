#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import datasets
import pandas as pd
from datasets import load_dataset

LABELS = ["Irrelevant", "Relevant"]


def load_privacy_qa(directory: str) -> datasets.DatasetDict:
    # load and process the train dataset
    train_df = pd.read_csv(os.path.join(directory, "policy_train.tsv"), sep="\t")
    train_df = train_df[["Query", "Segment", "Label"]].rename(
        columns={"Query": "question", "Segment": "text", "Label": "label"}
    )

    # collect information about label
    label_info = datasets.ClassLabel(names=LABELS)
    train_dataset = datasets.Dataset.from_pandas(train_df, preserve_index=False)

    # work on the test dataset
    test_df = pd.read_csv(os.path.join(directory, "policy_test.tsv"), sep="\t")
    test_df = test_df[["Query", "Segment", "Any_Relevant"]].rename(
        columns={"Query": "question", "Segment": "text", "Any_Relevant": "label"}
    )
    test_dataset = datasets.Dataset.from_pandas(test_df, preserve_index=False)

    # make split using HF datasets internal methods
    train_valid_dataset_dict = train_dataset.train_test_split(test_size=0.15, seed=42)

    # concatenate both datasets
    combined = datasets.DatasetDict(
        {
            "train": train_valid_dataset_dict["train"],
            "validation": train_valid_dataset_dict["test"],
            "test": test_dataset,
        }
    )

    # map labels to integers and add feature information
    for split in ["train", "validation", "test"]:
        combined[split] = combined[split].map(
            lambda examples: {
                "label": [label_info.str2int(label) for label in examples["label"]]
            },
            batched=True,
        )
        combined[split].features["label"] = label_info

    return combined


def privacy_qa_to_text2text(path='alzoubi36/privacy_qa'):
    # Load the dataset
    dataset_dict = load_dataset(path)
    # collect information about label
    label_info = datasets.ClassLabel(names=LABELS)

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        # Add prefix to each datapoint
        dataset = dataset.map(
            lambda example: {'text': f"privacy_qa question: {example['question']} text: {example['text']}",
                             'label': example['label']}, remove_columns=['question'])

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
                r"\privacy_qa"
    dataset_dict = load_privacy_qa(directory)
    dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
