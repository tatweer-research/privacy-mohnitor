#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
from typing import Dict, List, Tuple
from datasets import Dataset, load_dataset
import datasets


def expand_dataset_per_task(ds, tasks):
    # only one label per example, split the data into multiple tasks
    multi_datasets = {}
    for i, st in enumerate(tasks):
        per_task_dataset = {"tokens": [], "tags": [], "subtask": []}
        for example in ds:
            per_task_dataset["tokens"].append(example["tokens"])
            per_task_dataset["tags"].append([tag[i] for tag in example["tags"]])
            per_task_dataset["subtask"].append(st)
        multi_datasets[st] = Dataset.from_dict(per_task_dataset)
    return multi_datasets


SUBTASKS = sorted(["COLLECT", "NOT_COLLECT", "NOT_SHARE", "SHARE"])
LABELS = sorted(
    [
        ["COLLECT"],
        ["NOT_COLLECT"],
        ["NOT_SHARE"],
        ["SHARE"],
    ]
)


def read_conll_file(file_path: str) -> Dict[str, List[List[str]]]:
    # read all lines in CONLL file and strip trailing newlines
    with open(file_path, "r") as input_file_stream:
        conll_lines = [line.rstrip() for line in input_file_stream]

    # create global dictionary for storing data
    data: Dict[str, List[List[str]]] = {"tokens": [], "tags": []}

    # loop through lines in CONLL file
    for line in conll_lines:
        if line == "-DOCSTART- -X- O O":
            # skip line if DOCSTART encountered
            continue
        elif line == "":
            # append a new list as an empty string denotes
            # the completion of a single annotation
            data["tokens"].append([])
            data["tags"].append([])
        else:
            # in all other cases, split the line and append
            # one token and one NER tag to the final list
            token, tag = line.split(" _ _ ")
            data["tokens"][-1].append(token)
            data["tags"][-1].append(tag)

    return data


def merge_tags(tags: List[List[List[str]]]) -> List[List[Tuple[str, ...]]]:
    # perform a nested zip operation to combine token-level NER tags
    return [list(zip(*tag)) for tag in list(zip(*tags))]


def load_piextract(directory: str) -> datasets.DatasetDict:
    # define global data dictionary
    data: Dict[str, List[Dict[str, List[List[str]]]]] = {"train": [], "test": []}

    # define empty DatasetDict
    combined = datasets.DatasetDict()

    # loop over tasks and CONLL files associated per task
    for task in ["CollectUse_true", "CollectUse_false", "Share_false", "Share_true"]:
        for conll_file in glob(os.path.join(directory, task, "*.conll03")):
            if os.path.basename(conll_file).startswith("train"):
                split = "train"
            else:
                split = "test"

            # append parsed CONLL file to dictionary by split
            data[split].append(read_conll_file(conll_file))

    # loop over each data split
    for split, data_split in data.items():
        # flatten tokens from all four tasks in this split
        all_tokens = [data_split_subset["tokens"] for data_split_subset in data_split]

        # flatten NER tags from all four tasks in this split
        all_tags = [data_split_subset["tags"] for data_split_subset in data_split]

        # ensure that all tokens are exactly the same (assumption for merging)
        assert all([tokens == all_tokens[0] for tokens in all_tokens])

        # merge all NER tags
        merged_tags = merge_tags(all_tags)

        # convert dictionary into HF dataset and insert into DatasetDict
        combined[split] = datasets.Dataset.from_dict(
            {"tokens": all_tokens[0], "tags": merged_tags}
        )

    # make split using HF datasets internal methods
    train_valid_dataset_dict = combined["train"].train_test_split(
        test_size=0.15, seed=42
    )
    # reassign splits to combined and multiply tags to rows
    combined["train"] = expand_dataset_per_task(
        train_valid_dataset_dict["train"], SUBTASKS
    )
    combined["validation"] = expand_dataset_per_task(
        train_valid_dataset_dict["test"], SUBTASKS
    )

    combined["test"] = expand_dataset_per_task(combined["test"], SUBTASKS)

    # get all the unique tags and add to feature information
    label_names = {
        task: ["O"] + [f"{pre}-{label}" for pre in ["B", "I"] for label in tags]
        for task, tags in zip(SUBTASKS, LABELS)
    }
    for split in ["train", "validation", "test"]:
        for st in SUBTASKS:
            combined[split][st].features["tags"] = datasets.Sequence(
                feature=datasets.ClassLabel(names=label_names[st])
            )
    combined['train'] = Dataset.from_dict(combined['train'])
    combined['test'] = Dataset.from_dict(combined['test'])
    combined['validation'] = Dataset.from_dict(combined['validation'])

    return combined


def to_text2text(path='alzoubi36/piextract'):
    # Load the dataset
    dataset_dict = load_dataset(path)

    def generate_extra_ids(example: list):
        return [f'<extra_id_{i}>' for i in range(len(example))]

    def combine_subtasks_labels(example: dict):
        subtasks = example.keys()
        tags = (example[subtask]['tags'] for subtask in subtasks)
        combined_labels = map(list, zip(*tags))
        combined_labels = map(set, combined_labels)
        return list(combined_labels)

    # TODO: remove
    def check_condition(result):
        return 'B-NOT_COLLECT' in result and 'B-COLLECT' in result \
                    or 'I-NOT_COLLECT' in result and 'I-COLLECT' in result \
                    or 'I-NOT_SHARE' in result and 'I-SHARE' in result \
                    or 'B-NOT_SHARE' in result and 'B-SHARE' in result

    def check_overlapping_labels():
        dataset = dataset_dict['train']
        c = 0
        for i in dataset:
            c += len(i['COLLECT']['tokens'])
        print('dataset_tokens:', c)
        c_datapoints = 0
        c_overlapped_tokens = 0
        c_err = 0
        c_err_tokens = 0

        for example in dataset:
            results = combine_subtasks_labels(example)
            check_list = [len(x) > 2 for x in results]
            error_list = [check_condition(result) for result in results]
            if any(check_list):
                [print(result) for result in results if len(result) > 2]
                print(example)
                c_datapoints += 1
                c_overlapped_tokens += sum(check_list)
                c_err_tokens += sum(error_list)
                c_err += any(error_list)
                print()
        # print('datapoints_proportion:', c_datapoints/len(dataset))
        # print('c_datapoints:', c_datapoints)
        # print('c_err:', c_err)
        # print('c_err_tokens:', c_err_tokens)
        # print('c_overlapped_tokens:', c_overlapped_tokens)
        # print(len(dataset))
    check_overlapping_labels()
    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        temp_dataset = Dataset.from_dict({'data': dataset[SUBTASKS[0]] + dataset[SUBTASKS[1]] +
                                                  dataset[SUBTASKS[2]] + dataset[SUBTASKS[3]]})

        # Merge label columns into a single column
        dataset = temp_dataset
        dataset = dataset.map(lambda example: {'text': f"piextract "
                                                       f"sentence: {str(example['data']['tokens'])}",
                                               'label': str(example['data']['tags'])},
                              remove_columns=['data'])

        dataset_dict[split] = dataset

    return dataset_dict


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\piextract"
    # dataset_dict = load_piextract(directory)
    dataset_dict = to_text2text()
    # dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
