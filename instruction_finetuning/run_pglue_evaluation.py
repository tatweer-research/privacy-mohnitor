from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    FlaxT5ForConditionalGeneration,
)

import jax.numpy as jnp
import time
import json
from pathlib import Path

from instruction_finetuning.data_preprocessing import text2text_functions, from_text_functions
from instruction_finetuning.data_preprocessing import TASKS as PRIVACY_GLUE_TASKS

# from jax_smi import initialise_tracking
# initialise_tracking()

FLAX = False
PT = True


# some computation...
def initialize_model(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    models = {'flax': '', 'pt': ''}
    if FLAX:
        models['flax'] = FlaxT5ForConditionalGeneration.from_pretrained(model_name,
                                                               dtype=getattr(jnp, "float32"),
                                                               seed=42)
    if PT:
        models['pt'] = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=True).to('cpu')
    return tokenizer, models


def generate_model_outputs(models, tokenizer, input_text):
    results = {'flax': "", 'pt': ""}

    if FLAX:
        inputs = tokenizer(input_text,
                           padding="max_length",
                           max_length=512,
                           truncation=True,
                           return_attention_mask=False,
                           return_tensors="np"
                           )
        output_sequences = models['flax'].generate(
            input_ids=inputs["input_ids"],
            # attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_length=512,
        ).sequences
        results['flax'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
    if PT:
        inputs = tokenizer(input_text,
                           max_length=512,
                           padding="max_length",
                           truncation=True,
                           return_attention_mask=False,
                           return_tensors="pt"
                           )
        output_sequences = models['pt'].generate(
            input_ids=inputs["input_ids"],
            # attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_length=512,
        )

        results['pt'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results


def generate_model_outputs_dataset(models, tokenizer, pglue_task="policy_ie_a"):
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()

    DATASET = 'test'
    inputs = [dataset_dict[DATASET][i]['text'] for i in range(len(dataset_dict[DATASET]))]
    # inputs = [dataset_dict[DATASET][i]['text'] for i in range(20)]
    print("Generating model outputs for {} examples".format(len(inputs)))
    outputs = generate_model_outputs(models, tokenizer, inputs)
    results_path = Path(model_name).parent / "results.json"
    with open(results_path, 'w', encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

    for i in range(20):
        print("-" * 20 + "model input" + "-" * 20)
        input = inputs[i]
        print(input)
        print("-" * 20 + "original labels" + "-" * 20)
        labels = [dataset_dict[DATASET][i]['label']]
        print(labels)
        print("-" * 20 + "model output" + "-" * 20)
        # print("flax: ", outputs['flax'][i])
        print("pt: ", outputs['pt'][i])
        # print("-" * 20 + "model output from_text" + "-" * 20)
        # transformed = from_text_func(output[0], required_subtask="COLLECT")
        # print("number of labels: ", len(transformed))
        # print(transformed)
    print()


if __name__ == '__main__':
    task_prefix = "translate English to German: "
    # use different length sentences to test batching
    sentences = ["The house is wonderful.", "I like to work in NYC."]
    model_name = "/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/experiments/2023-09-03/policy_ie_a/best_model/"

    tokenizer, models = initialize_model(model_name)

    start = time.time()
    generate_model_outputs_dataset(models, tokenizer)
    end = time.time()

    print("Generation time: ", end - start)
