import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    FlaxT5ForConditionalGeneration,
)

from instruction_finetuning.data_preprocessing import text2text_functions
from tqdm import tqdm

FLAX = True
PT = False


def initialize_model(model_name, tokenizer_name, use_flax=FLAX, use_pt=PT):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    models = {'flax': '', 'pt': ''}
    if use_flax:
        models['flax'] = FlaxT5ForConditionalGeneration.from_pretrained(model_name,
                                                                        dtype=getattr(jnp, "float32"),
                                                                        seed=42)
    if use_pt:
        models['pt'] = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=True).to('cpu')
    return tokenizer, models


def generate_model_outputs_parallel(models, tokenizer, inputs, use_flax=FLAX, use_pt=PT, max_generation_length=512):
    results = {'flax': "", 'pt': ""}

    if use_flax:
        jit_generate = jax.jit(models['flax'].generate, static_argnames=["max_length", "do_sample"])
        output_sequences = jit_generate(input_ids=inputs["input_ids"], max_length=max_generation_length,
                                        do_sample=False).sequences

        results['flax'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
    if use_pt:
        output_sequences = models['pt'].generate(
            input_ids=inputs["input_ids"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_length=max_generation_length,
        )

        results['pt'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results


def generate_model_outputs_dataset_parallel(models,
                                            tokenizer,
                                            outputs_path='outputs.json',
                                            pglue_task="privacy_qa",
                                            batch_size=16,
                                            max_generation_length=512):
    if not batch_size:
        batch_size = jax.device_count() * 32
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()

    DATASET = 'test'
    use_flax = FLAX
    use_pt = PT

    inputs = [dataset_dict[DATASET][i]['text'] for i in range(len(dataset_dict[DATASET]))]
    number_inputs = len(inputs)
    print("Generating model outputs for {} examples".format(len(inputs)))

    # Split inputs into batches of a specified size (e.g., batch_size)
    all_outputs = {'flax': [], 'pt': []}
    if use_flax:
        inputs = tokenizer(inputs,
                           padding="max_length",
                           max_length=512,
                           truncation=True,
                           return_attention_mask=False,
                           return_tensors="np"
                           )
    elif use_pt:
        inputs = tokenizer(inputs,
                           max_length=512,
                           padding="max_length",
                           truncation=True,
                           return_attention_mask=False,
                           return_tensors="pt"
                           )
    else:
        exit(1)
    for i in tqdm(range(0, number_inputs, batch_size), desc='Progress:'):
        batch_inputs = {k: v[i:i + batch_size] for k, v in inputs.items()}
        outputs = generate_model_outputs_parallel(models, tokenizer, batch_inputs, max_generation_length)
        if outputs['flax']:
            all_outputs['flax'] += outputs['flax']
        if outputs['pt']:
            all_outputs['pt'] += outputs['pt']

        with open(outputs_path, 'w', encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=4)


def generate_model_outputs(models, tokenizer, input_texts, use_flax=FLAX, use_pt=PT, max_generation_length=512):
    results = {'flax': "", 'pt': ""}

    if use_flax:
        inputs = tokenizer(input_texts,
                           padding="max_length",
                           max_length=512,
                           truncation=True,
                           return_attention_mask=False,
                           return_tensors="np"
                           )
        output_sequences = models['flax'].generate(
            input_ids=inputs["input_ids"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_length=max_generation_length,
        ).sequences
        results['flax'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
    if use_pt:
        inputs = tokenizer(input_texts,
                           max_length=512,
                           padding="max_length",
                           truncation=True,
                           return_attention_mask=False,
                           return_tensors="pt"
                           )
        output_sequences = models['pt'].generate(
            input_ids=inputs["input_ids"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_length=max_generation_length,
        )

        results['pt'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results


def generate_model_outputs_dataset(models,
                                   tokenizer,
                                   outputs_path='outputs.json',
                                   pglue_task="privacy_qa",
                                   batch_size=16,
                                   max_generation_length=512,
                                   split='test',
                                   examples_limit=None):
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()

    DATASET = split

    inputs = [dataset_dict[DATASET][i]['text'] for i in range(len(dataset_dict[DATASET]))][:examples_limit]

    print("Generating model outputs for {} examples".format(len(inputs)))

    # Split inputs into batches of a specified size (e.g., batch_size)
    all_outputs = {'flax': [], 'pt': []}
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        outputs = generate_model_outputs(models, tokenizer, batch_inputs,
                                         max_generation_length=max_generation_length)
        if outputs['flax']:
            all_outputs['flax'] += outputs['flax']
        if outputs['pt']:
            all_outputs['pt'] += outputs['pt']

        with open(outputs_path, 'w', encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=4)


def generate_and_evaluate(model_name="alzoubi36/pglue_piextract_priva_t5-base",
                          tokenizer_name="t5-small",
                          pglue_task="piextract",
                          output_json="outputs.json",
                          batch_size=16,
                          max_generation_length=512,
                          model=None,
                          tokenizer=None,
                          split="test",
                          examples_limit=None):
    if not model and not tokenizer:
        tokenizer, models = initialize_model(model_name, tokenizer_name)
    else:
        models = {"flax": model, "pt": ""}

    start = time.time()
    generate_model_outputs_dataset(models,
                                   tokenizer,
                                   batch_size=batch_size,
                                   max_generation_length=max_generation_length,
                                   pglue_task=pglue_task,
                                   outputs_path=output_json,
                                   split=split,
                                   examples_limit=examples_limit)
    end = time.time()
    print("Generation time: ", end - start)
    from instruction_finetuning.models_evaluation.calculate_f1 import TAKS_EVALUATION_FUNCTIONS
    evaluation_result = TAKS_EVALUATION_FUNCTIONS[pglue_task](output_json, split=split, examples_limit=examples_limit)
    print("Evaluation result: ", evaluation_result)
    return evaluation_result


if __name__ == '__main__':
    model_name = "/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/experiments/2023-09-18/policy_ie_a/latest_model/"

    generate_and_evaluate(model_name=model_name,
                          tokenizer_name=model_name,
                          pglue_task="policy_ie_a",
                          split="test",
                          output_json="outputs.json",
                          examples_limit=None,
                          batch_size=24,
                          max_generation_length=512)
