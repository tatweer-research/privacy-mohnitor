import os
from typing import cast, Dict

import datasets
from datasets import Dataset, load_dataset

SUBTASKS = ["type-I", "type-II"]
LABELS = [
    [
        "data-protector",
        "data-protected",
        "data-collector",
        "data-collected",
        "data-receiver",
        "data-retained",
        "data-holder",
        "data-provider",
        "data-sharer",
        "data-shared",
        "storage-place",
        "retention-period",
        "protect-against",
        "action",
    ],
    [
        "purpose-argument",
        "polarity",
        "method",
        "condition-argument",
    ],
]


def policy_ie_file_mapping(directory: str, filename: str) -> Dict[str, str]:
    # define patterns for file loading
    files = {}
    files["train"] = os.path.join(directory, "train", filename)
    files["validation"] = os.path.join(directory, "valid", filename)
    files["test"] = os.path.join(directory, "test", filename)
    return files


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


def load_policy_ie_b(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # load tokens which are common for all sub-tasks
    tokens = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq.in")
    ).map(lambda example: {"tokens": example["text"].split()}, remove_columns=["text"])

    # since this is task B, load all NER tags
    ner_tags_first = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq_type_I.out")
    ).map(
        lambda example: {"ner_tags_type_one": example["text"].split()},
        remove_columns=["text"],
    )
    ner_tags_second = datasets.load_dataset(
        "text", data_files=policy_ie_file_mapping(directory, "seq_type_II.out")
    ).map(
        lambda example: {"ner_tags_type_two": example["text"].split()},
        remove_columns=["text"],
    )

    # mypy-related fixes
    tokens = cast(datasets.DatasetDict, tokens)
    ner_tags_first = cast(datasets.DatasetDict, ner_tags_first)
    ner_tags_second = cast(datasets.DatasetDict, ner_tags_second)

    # zip together data in splits
    for split in ["train", "validation", "test"]:
        combined[split] = datasets.concatenate_datasets(
            [tokens[split], ner_tags_first[split], ner_tags_second[split]], axis=1
        )

    # merge NER tags and drop old ones
    combined = combined.map(
        lambda x: {"tags": list(zip(x["ner_tags_type_one"], x["ner_tags_type_two"]))},
        remove_columns=["ner_tags_type_one", "ner_tags_type_two"],
    )

    # reassign splits to combined and multiply tags to rows
    combined["train"] = expand_dataset_per_task(combined["train"], SUBTASKS)
    combined["validation"] = expand_dataset_per_task(combined["test"], SUBTASKS)

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


def policy_ie_b_to_text2text(path='alzoubi36/policy_ie_b'):
    # Load the dataset
    dataset_dict = load_dataset(path)

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        temp_dataset = Dataset.from_dict({'data': dataset[SUBTASKS[0]] + dataset[SUBTASKS[1]]})

        # Merge label columns into a single column
        dataset = temp_dataset
        dataset = dataset.map(lambda example: {
            'text': f"policy_ie_b {example['data']['subtask']} tokens: {str(example['data']['tokens'])}",
            'label': str(example['data']['tags'])},
                              remove_columns=['data'])

        dataset_dict[split] = dataset

    return dataset_dict


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\policy_ie_b"
    dataset_dict = load_policy_ie_b(directory)
    # dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    print()
