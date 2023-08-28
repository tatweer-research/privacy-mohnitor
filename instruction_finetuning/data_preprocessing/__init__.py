from instruction_finetuning.data_preprocessing.policy_ie_a import to_text2text as policy_ie_a
from instruction_finetuning.data_preprocessing.opp_115 import to_text2text as opp_115
from instruction_finetuning.data_preprocessing.piextract import to_text2text as piextract
from instruction_finetuning.data_preprocessing.policy_detection import to_text2text as policy_detection
from instruction_finetuning.data_preprocessing.policy_ie_b import to_text2text as policy_ie_b
from instruction_finetuning.data_preprocessing.policy_qa import to_text2text as policy_qa
from instruction_finetuning.data_preprocessing.privacy_qa import to_text2text as privacy_qa

from instruction_finetuning.data_preprocessing.policy_ie_a import label_from_text as policy_ie_a_from_text
from instruction_finetuning.data_preprocessing.opp_115 import label_from_text as opp_115_from_text
from instruction_finetuning.data_preprocessing.piextract import label_from_text as piextract_from_text
from instruction_finetuning.data_preprocessing.policy_detection import label_from_text as policy_detection_from_text
from instruction_finetuning.data_preprocessing.policy_ie_b import label_from_text as policy_ie_b_from_text
from instruction_finetuning.data_preprocessing.policy_qa import label_from_text as policy_qa_from_text
from instruction_finetuning.data_preprocessing.privacy_qa import label_from_text as privacy_qa_from_text

text2text_functions = {
    "privacy_glue": {
        "policy_ie_a": policy_ie_a,
        "opp_115": opp_115,
        "piextract": piextract,
        "policy_detection": policy_detection,
        "policy_ie_b": policy_ie_b,
        "policy_qa":  policy_qa,
        "privacy_qa": privacy_qa,
    }
}

from_text_functions = {
    "privacy_glue": {
        "policy_ie_a": policy_ie_a_from_text,
        "opp_115": opp_115_from_text,
        "piextract": piextract_from_text,
        "policy_detection": policy_detection_from_text,
        "policy_ie_b": policy_ie_b_from_text,
        "policy_qa":  policy_qa_from_text,
        "privacy_qa": privacy_qa_from_text,
    }
}

TASKS = list(text2text_functions["privacy_glue"].keys())
