from instruction_finetuning.data_preprocessing.policy_ie_a import to_text2text as policy_ie_a
from instruction_finetuning.data_preprocessing.opp_115 import to_text2text as opp_115
from instruction_finetuning.data_preprocessing.piextract import to_text2text as piextract
from instruction_finetuning.data_preprocessing.policy_detection import to_text2text as policy_detection
from instruction_finetuning.data_preprocessing.policy_ie_b import to_text2text as policy_ie_b
from instruction_finetuning.data_preprocessing.policy_qa import to_text2text as policy_qa
from instruction_finetuning.data_preprocessing.privacy_qa import to_text2text as privacy_qa

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

TASKS = list(text2text_functions["privacy_glue"].keys())
