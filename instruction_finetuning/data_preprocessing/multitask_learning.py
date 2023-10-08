import datasets
import numpy as np
import random

from datasets import concatenate_datasets

from instruction_finetuning.data_preprocessing.policy_ie_a import to_text2text as policy_ie_a
from instruction_finetuning.data_preprocessing.opp_115 import to_text2text as opp_115
from instruction_finetuning.data_preprocessing.piextract import to_text2text as piextract
from instruction_finetuning.data_preprocessing.policy_detection import to_text2text as policy_detection
from instruction_finetuning.data_preprocessing.policy_ie_b import to_text2text as policy_ie_b
from instruction_finetuning.data_preprocessing.policy_qa import to_text2text as policy_qa
from instruction_finetuning.data_preprocessing.privacy_qa import to_text2text as privacy_qa
from instruction_finetuning.data_preprocessing.title_generation import to_text2text as title_generation

text2text_functions = {
    "privacy_glue": {
        "policy_ie_a": policy_ie_a,
        "opp_115": opp_115,
        "piextract": piextract,
        "policy_detection": policy_detection,
        "policy_ie_b": policy_ie_b,
        "policy_qa":  policy_qa,
        "privacy_qa": privacy_qa,
        "title_generation": title_generation,
    }
}


def get_required_number_of_examples(sampling_rate, total_examples):
    return int(sampling_rate * total_examples)


def scale_dataset(dataset, requested_number_of_examples):
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=random.randint(1, 1000))

    if requested_number_of_examples <= len(dataset):
        # Take the first requested_number_of_examples examples
        scaled_dataset = dataset.select(range(requested_number_of_examples))
    else:
        # Repeat the dataset to reach the requested number of examples
        num_repeats = requested_number_of_examples // len(dataset)
        dataset_replicas = []
        for i in range(num_repeats):
            dataset_replicas.append(dataset)

        if len(dataset) * (num_repeats + 1) > requested_number_of_examples:
            remaining_examples = len(dataset) - (len(dataset) * (num_repeats + 1) - requested_number_of_examples)
            dataset_replicas.append(dataset.select(range(remaining_examples)))
        scaled_dataset = concatenate_datasets(dataset_replicas)

    return scaled_dataset.shuffle(seed=random.randint(1, 1000))


def prepare_multitask_dataset(alpha, split, percentage_of_total_examples=1.0):
    datasets = {}
    datasets_lengths = {}
    probabilities = {}
    sampling_rates = {}
    for task, func in text2text_functions["privacy_glue"].items():
        datasets[task] = func()[split]
        datasets_lengths[task] = len(datasets[task])

    total_examples = sum([length for length in datasets_lengths.values()])
    requested_total_examples = percentage_of_total_examples * total_examples

    for task, length in datasets_lengths.items():
        probabilities[task] = length / total_examples

    for task, prob in probabilities.items():
        sampling_rates[task] = np.power(probabilities[task], alpha) / np.sum([np.power(probabilities[key], alpha)
                                                                              for key in probabilities.keys()])
    scaled_datasets = []
    for task, dataset in datasets.items():
        requested_number_of_examples = get_required_number_of_examples(sampling_rates[task], requested_total_examples)
        scaled_dataset = scale_dataset(dataset, requested_number_of_examples)
        scaled_datasets.append(scaled_dataset)

    multitask_dataset = concatenate_datasets(scaled_datasets)

    return multitask_dataset.shuffle(seed=random.randint(1, 1000))


def prepare_multitask_datasetdict(alpha=0.001, percentage_of_total_examples=0.5):
    dataset_dict = datasets.DatasetDict()

    splits = ['train', 'test', 'validation']
    for split in splits:
        dataset_dict[split] = prepare_multitask_dataset(alpha=alpha,
                                                        split=split,
                                                        percentage_of_total_examples=percentage_of_total_examples)
    return dataset_dict


if __name__ == '__main__':
    result = prepare_multitask_datasetdict()
    print()
