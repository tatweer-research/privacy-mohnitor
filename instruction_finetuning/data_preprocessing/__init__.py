from instruction_finetuning.data_preprocessing.policy_ie_a import label_from_text as policy_ie_a_from_text
from instruction_finetuning.data_preprocessing.opp_115 import label_from_text as opp_115_from_text
from instruction_finetuning.data_preprocessing.piextract import label_from_text as piextract_from_text
from instruction_finetuning.data_preprocessing.policy_detection import label_from_text as policy_detection_from_text
from instruction_finetuning.data_preprocessing.policy_ie_b import label_from_text as policy_ie_b_from_text
from instruction_finetuning.data_preprocessing.policy_qa import label_from_text as policy_qa_from_text
from instruction_finetuning.data_preprocessing.privacy_qa import label_from_text as privacy_qa_from_text
from instruction_finetuning.data_preprocessing.multitask_learning import text2text_functions, \
    prepare_multitask_datasetdict

from instruction_finetuning.data_preprocessing.policy_ie_a import flan_text2text as policy_ie_a_flan
from instruction_finetuning.data_preprocessing.opp_115 import flan_text2text as opp_115_flan
from instruction_finetuning.data_preprocessing.piextract import flan_text2text as piextract_flan
from instruction_finetuning.data_preprocessing.policy_detection import flan_text2text as policy_detection_flan
from instruction_finetuning.data_preprocessing.policy_ie_b import flan_text2text as policy_ie_b_flan
from instruction_finetuning.data_preprocessing.policy_qa import flan_text2text as policy_qa_flan
from instruction_finetuning.data_preprocessing.privacy_qa import flan_text2text as privacy_qa_flan
from instruction_finetuning.data_preprocessing.title_generation import flan_text2text as title_generation_flan

text2text_functions['multitask'] = prepare_multitask_datasetdict

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

flan_text2text_functions = {
    "privacy_glue": {
        "policy_ie_a": policy_ie_a_flan,
        "opp_115": opp_115_flan,
        "piextract": piextract_flan,
        "policy_detection": policy_detection_flan,
        "policy_ie_b": policy_ie_b_flan,
        "policy_qa":  policy_qa_flan,
        "privacy_qa": privacy_qa_flan,
        "title_generation": title_generation_flan,
    }
}


TASKS = list(text2text_functions["privacy_glue"].keys()) + ["multitask"]
