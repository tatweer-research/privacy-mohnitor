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


def generate_extra_ids(example: list):
    """Generate extra ids for each token in one example."""
    return [f'<token_id_{i}>' for i in range(len(example['type-I']['tokens']))]


def combine_subtasks_labels(example: dict):
    """Combines labels from all subtasks into one label. Labels are separated by a dot."""
    subtasks = example.keys()
    tags = (example[subtask]['tags'] for subtask in subtasks)
    combined_labels = map(list, zip(*tags))
    combined_labels = map(set, combined_labels)

    def filter_labels(labels):
        if len(labels) > 1:
            try:
                labels.remove('O')
            except KeyError:
                pass
        return labels

    combined_labels = map(filter_labels, combined_labels)
    combined_labels = map('&&'.join, combined_labels)
    example['type-I']['tags'] = list(combined_labels)
    return example


def add_extra_ids(example, mode='tags', subtask='type-I'):
    """Adds extra ids to each token in one example."""
    extra_ids = generate_extra_ids(example)
    transformed = []
    for id, token in zip(extra_ids, example[subtask][mode]):
        transformed += [' ' + id + ' ' + token]
    return transformed


def to_text2text(path='alzoubi36/policy_ie_b', subtask: str = 'combined'):
    """Converts the piextract dataset to a text2text dataset."""

    # Load the dataset
    dataset_dict = load_dataset(path)

    subtasks = SUBTASKS

    if subtask != 'combined' and subtask not in subtasks:
        raise ValueError(f"subtask must be one of {subtasks} or 'combined'")

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        if subtask == 'combined':
            dataset = dataset.map(lambda example: {'text': f"policy_ie_b {subtask} "
                                                           f"sentence:{''.join(add_extra_ids(combine_subtasks_labels(example), mode='tokens'))}",
                                                   'label': ''.join(add_extra_ids(example, mode='tags'))},
                                  remove_columns=subtasks)
            dataset_dict[split] = dataset
        else:
            dataset = dataset.map(lambda example: {'text': f"policy_ie_b {subtask} "
                                                           f"sentence:{''.join(add_extra_ids(example, mode='tokens', subtask=subtask))}",
                                                   'label': ''.join(
                                                       add_extra_ids(example, mode='tags', subtask=subtask))},
                                  remove_columns=subtasks)
            dataset_dict[split] = dataset

    return dataset_dict


def split_labels(label_list, required_subtask='type-I'):
    all_subtask_labels = LABELS[SUBTASKS.index(required_subtask)]
    for i, label in enumerate(label_list):
        boolean_list = []
        for subtask_label in all_subtask_labels:
            subtask_label = '-' + subtask_label
            boolean_list.append(subtask_label in label)
            if subtask_label in label:
                temp = label.split('&&')
                temp = filter(lambda x: subtask_label in x, temp)
                label_list[i] = next(temp)
                break
        if not any(boolean_list):
            label_list[i] = 'O'
    return label_list


def label_from_text(label, required_subtask, mode='combined'):
    subtasks = SUBTASKS
    if required_subtask not in subtasks:
        raise ValueError(f"required_subtask must be one of {subtasks}")

    if mode == 'combined':
        labels = label.split()[1::2]  # Splitting and selecting odd-indexed elements
        return split_labels(labels, required_subtask=required_subtask)
    else:
        return label.split()[1::2]


def label_from_text_functionality_checker():
    from tqdm import tqdm

    SPLIT = 'test'
    SUBTASK = SUBTASKS[1]
    dataset = load_dataset(path='alzoubi36/policy_ie_b')[SPLIT]
    tansformed_dataset = to_text2text()[SPLIT]
    for i in tqdm(range(len(dataset))):
        example = dataset[SUBTASK][i]['tags']
        transformed_example = label_from_text(tansformed_dataset[i]['label'], SUBTASK)
        assert transformed_example == example, f"Example {i} is not equal"
        print()


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\policy_ie_b"
    # dataset_dict = load_policy_ie_b(directory)
    dataset_dict = to_text2text()
    # dataset_dict.push_to_hub('alzoubi36/policy_ie_a')
    # label_from_text_functionality_checker()
    print()
