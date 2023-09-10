import json
import time
from pathlib import Path

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
            # attention_mask=inputs["attention_mask"],
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
            # attention_mask=inputs["attention_mask"],
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
                                   max_generation_length=512):
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()

    DATASET = 'test'

    inputs = [dataset_dict[DATASET][i]['text'] for i in range(len(dataset_dict[DATASET]))]
    print("Generating model outputs for {} examples".format(len(inputs)))

    # Split inputs into batches of a specified size (e.g., batch_size)
    all_outputs = {'flax': [], 'pt': []}
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        outputs = generate_model_outputs(models, tokenizer, batch_inputs, max_generation_length=max_generation_length)
        if outputs['flax']:
            all_outputs['flax'] += outputs['flax']
        if outputs['pt']:
            all_outputs['pt'] += outputs['pt']

        with open(outputs_path, 'w', encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    model_name = "alzoubi36/pglue_privacy_qa_priva_t5-small"

    tokenizer, models = initialize_model(model_name, "t5-small")

    start = time.time()
    generate_model_outputs_dataset(models, tokenizer, max_generation_length=5)
    end = time.time()

    print("Generation time: ", end - start)
