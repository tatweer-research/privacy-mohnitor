from sklearn.metrics import classification_report, f1_score
from datasets import load_dataset
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support

from instruction_finetuning.data_preprocessing.policy_ie_a import label_from_text as policy_ie_a_from_text
from instruction_finetuning.data_preprocessing.policy_ie_a import LABELS as policy_ie_a_labels

from instruction_finetuning.data_preprocessing.opp_115 import label_from_text as opp_115_from_text
from instruction_finetuning.data_preprocessing.opp_115 import LABELS as opp_115_labels

from instruction_finetuning.data_preprocessing.privacy_qa import label_from_text as privacy_qa_from_text
from instruction_finetuning.data_preprocessing.privacy_qa import LABELS as privacy_qa_labels

from instruction_finetuning.data_preprocessing.piextract import label_from_text as piextract_from_text
from instruction_finetuning.data_preprocessing.piextract import SUBTASKS as piextract_subtasks

from instruction_finetuning.data_preprocessing.policy_ie_b import label_from_text as policy_ie_b_from_text
from instruction_finetuning.data_preprocessing.policy_ie_b import SUBTASKS as policy_ie_b_subtasks
from instruction_finetuning.data_preprocessing.policy_ie_b import LABELS as policy_ie_b_labels


def evaluate_policy_detection(model_outputs_path, split='test'):
    # Dataset
    dataset_dict = load_dataset('alzoubi36/policy_detection')
    dataset = dataset_dict[split]

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [example['label'] for example in dataset]
    y_pred = [1 if example == 'policy' else 0 for example in results['flax']]
    target_names = ['not_policy', 'policy']
    result_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    result = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    f1_score_result = f1_score(y_true, y_pred, average='micro', labels=list(range(2)))
    return f1_score_result


def evaluate_policy_ie_a(model_outputs_path, split='test'):
    # Dataset
    dataset_dict = load_dataset('alzoubi36/policy_ie_a')
    dataset = dataset_dict[split]

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [example['label'] for example in dataset]
    y_pred = [policy_ie_a_from_text(example) for example in results['flax']]
    combined_y_pred = []
    for i in y_pred:
        combined_y_pred.extend(i)
    y_pred = combined_y_pred
    target_names = policy_ie_a_labels
    result_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    result = classification_report(y_true, y_pred, target_names=target_names, digits=4, labels=[0, 1, 2, 3, 4])
    f1_score_result = f1_score(y_true, y_pred, average='micro', labels=list(range(4)))
    return f1_score_result


def evaluate_opp_115(model_outputs_path, split='test'):
    # Dataset
    dataset_dict = load_dataset('alzoubi36/opp_115')
    dataset = dataset_dict[split]

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [example['label'] for example in dataset]
    y_pred = [opp_115_from_text(example) for example in results['flax']]

    # TODO: find a better way to handle empty predictions or labels with zero predictions
    for i in range(len(y_pred)):
        if not y_pred[i]:
            y_pred[i] = [7]
        break
    y_true = MultiLabelBinarizer().fit_transform(y_true)
    y_pred = MultiLabelBinarizer().fit_transform(y_pred)

    target_names = opp_115_labels
    result_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    result = classification_report(y_true, y_pred, target_names=target_names, digits=4, labels=list(range(12)))
    f1_score_result = f1_score(y_true, y_pred, average='micro', labels=list(range(12)))
    return f1_score_result


def evaluate_privacy_qa(model_outputs_path, split='test'):
    # Dataset
    dataset_dict = load_dataset('alzoubi36/privacy_qa')
    dataset = dataset_dict[split]

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [example['label'] for example in dataset]
    y_pred = [privacy_qa_from_text(example) for example in results['flax']]
    target_names = privacy_qa_labels
    result_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    result = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    f1_score_result = f1_score(y_true, y_pred, average='micro', labels=list(range(len(target_names))))
    return f1_score_result


def evaluate_piextract(model_outputs_path, split='test'):
    # Dataset
    dataset_dict = load_dataset('alzoubi36/piextract')
    dataset = dataset_dict[split]

    def postprocess(y_true, y_pred):
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if len(true) > len(pred):
                pred.extend(['O'] * (len(true) - len(pred)))
            elif len(true) < len(pred):
                pred = pred[:len(true)]
            assert len(true) == len(pred)
            pred = [example.replace('B-', '').replace('I-', '') for example in pred]
            true = [example.replace('B-', '').replace('I-', '') for example in true]
            y_true[i] = true
            y_pred[i] = pred
        return y_true, y_pred

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)
    f1_scores = []
    for subtask in piextract_subtasks:
        y_true = [example[subtask]['tags'] for example in dataset]
        y_pred = [piextract_from_text(example, required_subtask=subtask) for example in results['flax']]
        y_true, y_pred = postprocess(y_true, y_pred)
        _, _, f1_score_result, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        f1_scores.append(f1_score_result)
        print(f'{subtask}:\n{f1_score_result}\n')
    return sum(f1_scores) / len(f1_scores)


def evaluate_policy_ie_b(model_outputs_path, split='test'):
    # Dataset
    dataset_dict = load_dataset('alzoubi36/policy_ie_b')
    dataset = dataset_dict[split]

    def postprocess(y_true, y_pred):
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            if len(true) > len(pred):
                pred.extend(['O'] * (len(true) - len(pred)))
            elif len(true) < len(pred):
                pred = pred[:len(true)]
            assert len(true) == len(pred)
            pred = [example.split('.')[0].replace('B-', '').replace('I-', '') for example in pred]
            true = [example.split('.')[0].replace('B-', '').replace('I-', '') for example in true]
            y_true[i] = true
            y_pred[i] = pred

        # return combined_y_true, combined_y_pred
        return y_true, y_pred

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)
    f1_scores = []
    for i, subtask in enumerate(policy_ie_b_subtasks):
        y_true = [example[subtask]['tags'] for example in dataset]
        y_pred = [policy_ie_b_from_text(example, required_subtask=subtask) for example in results['flax']]
        y_true, y_pred = postprocess(y_true, y_pred)
        _, _, f1_score_result, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        f1_scores.append(f1_score_result)
        print(f'{subtask}:\n{f1_score_result}\n')
    return sum(f1_scores) / len(f1_scores)


def levenshtein_distance(str1, str2):
    # Create a matrix to store the distances
    rows = len(str1) + 1
    cols = len(str2) + 1
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    # Initialize the first row and column
    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j

    # Fill in the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # Deletion
                matrix[i][j - 1] + 1,  # Insertion
                matrix[i - 1][j - 1] + cost,  # Substitution
            )

    # The final value in the matrix represents the Levenshtein distance
    return matrix[rows - 1][cols - 1]


def evaluate_policy_qa(model_outputs_path, split='test'):
    def accuracy(y_true, y_pred):
        levenshtein_distances = []
        for str1, str2 in zip(y_true, y_pred):
            levenshtein_distances.append(levenshtein_distance(str1, str2))
        levenshtein_distances = np.array(levenshtein_distances)
        min_value = np.min(levenshtein_distances)
        max_value = np.max(levenshtein_distances)

        # Normalize the array
        normalized_array = (levenshtein_distances - min_value) / (max_value - min_value)
        binary_array = np.where(normalized_array < 0.2, 1, 0)
        return np.sum(binary_array) / len(binary_array)

    dataset_dict = load_dataset('alzoubi36/policy_qa')
    dataset = dataset_dict[split]

    # Evaluation results
    with open(model_outputs_path,
              'r', encoding='utf-8') as f:
        results = json.load(f)

    y_true = [example['answers']['text'] for example in dataset]
    y_pred = [example for example in results['flax']]
    return accuracy(y_true, y_pred)


TAKS_EVALUATION_FUNCTIONS = {"policy_ie_a": evaluate_policy_ie_a,
                             "opp_115": evaluate_opp_115,
                             "piextract": evaluate_piextract,
                             "policy_detection": evaluate_policy_detection,
                             "policy_ie_b": evaluate_policy_ie_b,
                             "policy_qa": evaluate_policy_qa,
                             "privacy_qa": evaluate_privacy_qa}

if __name__ == '__main__':
    print(evaluate_piextract('outputs.json'))
