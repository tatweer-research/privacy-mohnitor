# Used this as reference for generate parallelization https://github.com/huggingface/transformers/issues/20794#issuecomment-1483719000

import json
import time
from pathlib import Path

import numpy as np
# from jax_smi import initialise_tracking
import jax
import jax.numpy as jnp
from flax.training.common_utils import shard
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    FlaxT5ForConditionalGeneration,
)

from instruction_finetuning.data_preprocessing import text2text_functions
from tqdm import tqdm

# initialise_tracking(dir_prefix='/home/Mohammad.Al-Zoubi/jax-smi')

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


def generate_model_outputs(models, inputs, use_flax=FLAX, use_pt=PT):
    results = {'flax': "", 'pt': ""}

    # def generate_step(input_ids):
    #     output_sequences = models['flax'].generate(
    #         input_ids=input_ids,
    #         # attention_mask=inputs["attention_mask"],
    #         do_sample=False,  # disable sampling to test if batching affects output
    #         max_length=512,
    #     )
    #     return output_sequences.sequences

    if use_flax:
        jit_generate = jax.jit(models['flax'].generate, static_argnames=["max_length", "do_sample"])
        output_sequences = jit_generate(input_ids=inputs["input_ids"], max_length=512, do_sample=False).sequences

        # model_inputs = shard(inputs["input_ids"])
        # output_sequences = jax.pmap(generate_step)(model_inputs).tolist()
        # output_sequences = np.asarray(output_sequences)
        # output_sequences = output_sequences.reshape(output_sequences.shape[0]*output_sequences.shape[1], output_sequences.shape[2])
        results['flax'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
    if use_pt:
        output_sequences = models['pt'].generate(
            input_ids=inputs["input_ids"],
            # attention_mask=inputs["attention_mask"],
            do_sample=False,  # disable sampling to test if batching affects output
            max_length=512,
        )

        results['pt'] = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    return results


def generate_model_outputs_dataset(models, tokenizer, outputs_path='outputs.json', pglue_task="policy_ie_b",
                                   batch_size=None):
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
        outputs = generate_model_outputs(models, batch_inputs)
        if outputs['flax']:
            all_outputs['flax'] += outputs['flax']
        if outputs['pt']:
            all_outputs['pt'] += outputs['pt']

        with open(outputs_path, 'w', encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    model_name = "alzoubi36/pglue_opp_115_priva_t5-small"

    tokenizer, models = initialize_model(model_name, "t5-small")

    start = time.time()
    generate_model_outputs_dataset(models, tokenizer)
    end = time.time()

    print("Generation time: ", end - start)
