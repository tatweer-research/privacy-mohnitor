import os
import datasets
import pandas as pd
from datasets import load_dataset

LABELS = ["Not Policy", "Policy"]


def load_policy_detection(directory: str) -> datasets.DatasetDict:
    # initialize DatasetDict object
    combined = datasets.DatasetDict()

    # read csv file and choose subset of columns
    df = pd.read_csv(os.path.join(directory, "1301_dataset.csv"), index_col=0)
    df = df[["policy_text", "is_policy"]]

    # replace labels from boolean to strings for consistency
    df["is_policy"] = df["is_policy"].replace({True: "Policy", False: "Not Policy"})

    # rename columns for consistency
    df = df.rename(columns={"policy_text": "text", "is_policy": "label"})

    # convert into HF datasets
    dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
    label_info = datasets.ClassLabel(names=LABELS)

    # make split using HF datasets internal methods
    train_test_dataset_dict = dataset.train_test_split(test_size=0.3, seed=42)
    train_valid_dataset_dict = train_test_dataset_dict["train"].train_test_split(
        test_size=0.15, seed=42
    )

    # manually assign them to another DatasetDict
    combined["train"] = train_valid_dataset_dict["train"]
    combined["validation"] = train_valid_dataset_dict["test"]
    combined["test"] = train_test_dataset_dict["test"]

    # collect and distribute information about label
    for split in ["train", "validation", "test"]:
        combined[split] = combined[split].map(
            lambda examples: {
                "label": [label_info.str2int(label) for label in examples["label"]]
            },
            batched=True,
        )
        combined[split].features["label"] = label_info

    return combined


def to_text2text(path='alzoubi36/policy_detection'):
    """Convert policy_detection dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path)

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        # Add prefix to each datapoint
        dataset = dataset.map(lambda example: {'text': f"policy detection: {example['text']}",
                                               'label': example['label']})

        # Transform labels
        label_mapping = {1: 'policy', 0: 'not policy'}
        dataset = dataset.map(lambda example: {'text': example['text'],
                                               'label': label_mapping[example['label']]})
        dataset_dict[split] = dataset

    return dataset_dict


def flan_text2text(path='alzoubi36/policy_detection'):
    """Convert policy_detection dataset to text2text format"""

    # Load the dataset
    dataset_dict = load_dataset(path)

    for split in dataset_dict.keys():
        dataset = dataset_dict[split]
        # Add prefix to each datapoint
        dataset = dataset.map(lambda example: {'text': f"""Answer the following question: Is this text a privacy policy? Give the labels either "policy" or "not_policy" Text: {example['text']}""",
                                               'label': example['label']})

        # Transform labels
        label_mapping = {1: 'policy', 0: 'not policy'}
        dataset = dataset.map(lambda example: {'text': example['text'],
                                               'label': label_mapping[example['label']]})
        dataset_dict[split] = dataset

    return dataset_dict


def label_from_text(label):
    label_mapping = {'policy': 1, 'not policy': 0}
    return label_mapping[label]


if __name__ == "__main__":
    directory = r"C:\Users\Mohammad.Al-zoubi\Documents\projects\privacy-mohnitor\instruction_finetuning\data" \
                r"\policy_detection"
    # dataset_dict = load_policy_detection(directory)
    dataset_dict = to_text2text()
    # dataset_dict.push_to_hub('alzoubi36/policy_detection')
    print()
