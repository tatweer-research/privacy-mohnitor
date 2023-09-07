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

FLAX = False
PT = True


def initialize_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    models = {'flax': '', 'pt': ''}
    if FLAX:
        models['flax'] = FlaxT5ForConditionalGeneration.from_pretrained(model_name,
                                                                        dtype=getattr(jnp, "float32"),
                                                                        seed=42)
    if PT:
        models['pt'] = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=True).to('cpu')
    return tokenizer, models


def generate_model_outputs(models, tokenizer, input_texts):
    results = {'flax': "", 'pt': ""}

    if FLAX:
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
            max_length=512,
        ).sequences
        results['flax'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
    if PT:
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
            max_length=512,
        )

        results['pt'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results


def generate_model_outputs_dataset(models, tokenizer, pglue_task="opp_115"):
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()

    DATASET = 'test'
    inputs = [dataset_dict[DATASET][i]['text'] for i in range(len(dataset_dict[DATASET]))]
    print("Generating model outputs for {} examples".format(len(inputs)))

    # Split inputs into batches of a specified size (e.g., batch_size)
    batch_size = 32  # You can adjust this as needed
    all_outputs = {'flax': [], 'pt': []}
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch_inputs = inputs[i:i + batch_size]
        outputs = generate_model_outputs(models, tokenizer, batch_inputs)
        if outputs['flax']:
            all_outputs['flax'] += outputs['flax']
        if outputs['pt']:
            all_outputs['pt'] += outputs['pt']
        # Save or process the batch outputs here

        # Print the batch results
        # for j, input_text in enumerate(batch_inputs):
        #     print("-" * 20 + "Batch: {} | Example: {} ".format(i // batch_size, j) + "-" * 20)
        #     print("Input: ", input_text)
            # print("flax: ", outputs['flax'][j])
            # print("pt: ", outputs['pt'][j])
    results_path = Path(model_name).parent / "results.json"
    with open(results_path, 'w', encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=4)

    print()


if __name__ == '__main__':

    model_name = "/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/experiments/2023-09-03" \
                 "/opp_115/best_model/"

    tokenizer, models = initialize_model(model_name)

    start = time.time()
    generate_model_outputs_dataset(models, tokenizer)
    end = time.time()

    print("Generation time: ", end - start)
