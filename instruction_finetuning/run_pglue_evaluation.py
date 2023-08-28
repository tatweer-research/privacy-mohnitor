from transformers import (
    AutoTokenizer,
    FlaxT5ForConditionalGeneration,
)

from instruction_finetuning.data_preprocessing import text2text_functions, from_text_functions
from instruction_finetuning.data_preprocessing import TASKS as PRIVACY_GLUE_TASKS


def initialize_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def generate_model_outputs(model, tokenizer, input_text):
    inputs = tokenizer(input_text, return_tensors="jax", padding=True)
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
    )
    return tokenizer.batch_decode(output_sequences['sequences'].tolist(), skip_special_tokens=True)


def generate_model_outputs_dataset(model, tokenizer, pglue_task="piextract"):
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()
    tokenized = tokenize_datasets(dataset_dict, 512, tokenizer)
    tokens = tokenized['test'][0]['labels']
    from_text_func = from_text_functions["privacy_glue"][pglue_task]
    result = tokenizer.decode(tokens)

    print(result)
    print(dataset_dict['test'][0]['label'])
    print()


def tokenize_datasets(dataset, max_seq_length, tokenizer):
    # First we tokenize all the texts.
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # Since we make sure that all sequences are of the same length, no attention_mask is needed.
    def tokenize_function(examples):
        data = tokenizer(examples[text_column_name],
                         max_length=max_seq_length,
                         padding="max_length",
                         truncation=True,
                         return_attention_mask=False)
        data["labels"] = tokenizer(examples['label'],
                                   max_length=512,
                                   padding="max_length",
                                   truncation=True,
                                   return_attention_mask=False)["input_ids"]
        return data

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        # num_proc=self.data_args.preprocessing_num_workers,
        remove_columns=column_names,
    )

    return tokenized_datasets


if __name__ == '__main__':
    task_prefix = "translate English to German: "
    # use different length sentences to test batching
    sentences = ["The house is wonderful.", "I like to work in NYC."]
    model_name = "/home/Mohammad.Al-Zoubi/.cache/huggingface/hub/models--t5-small/snapshots" \
                 "/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/"

    tokenizer, model = initialize_model(model_name)
    # outputs = generate_model_outputs(model, tokenizer, [task_prefix + sentence for sentence in sentences])
    # print(outputs)
    generate_model_outputs_dataset(model, tokenizer)
