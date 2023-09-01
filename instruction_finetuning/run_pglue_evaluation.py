from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
)

from instruction_finetuning.data_preprocessing import text2text_functions, from_text_functions
from instruction_finetuning.data_preprocessing import TASKS as PRIVACY_GLUE_TASKS


def initialize_model(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")
    model = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=True)
    return tokenizer, model


def generate_model_outputs(model, tokenizer, input_text):
    inputs = tokenizer(input_text, padding="max_length",
                       max_length=512,
                       truncation=True,
                       return_attention_mask=False,
                       return_tensors="pt"
                       )
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
        max_length=512,
    )
    return tokenizer.batch_decode(output_sequences)


def generate_model_outputs_dataset(model, tokenizer, pglue_task="piextract"):
    dataset_dict = text2text_functions["privacy_glue"][pglue_task]()
    # tokenized = tokenize_datasets(dataset_dict, 512, tokenizer)
    # tokens = tokenized['test'][0]['labels']
    from_text_func = from_text_functions["privacy_glue"][pglue_task]
    # result = tokenizer.decode(tokens, skip_special_tokens=True)

    # print(dataset_dict['test'][0]['label'])
    # print(from_text_func(result, required_subtask="COLLECT"))
    DATASET = 'train'
    for i in range(20):
        print("-"*20 + "model input" + "-"*20)
        input = [dataset_dict[DATASET][i]['text']]
        print(input)
        print("-"*20 + "original labels" + "-"*20)
        labels = [dataset_dict[DATASET][i]['label']]
        print(labels)
        print("-"*20 + "model output" + "-"*20)
        output = generate_model_outputs(model, tokenizer, input)
        print(output)
        print("-"*20 + "model output from_text" + "-"*20)
        transformed = from_text_func(output[0], required_subtask="COLLECT")
        print("number of labels: ", len(transformed))
        print(transformed)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<COLLECT>", "<SHARE>", "<DELETE>"]})
    print()


if __name__ == '__main__':
    task_prefix = "translate English to German: "
    # use different length sentences to test batching
    sentences = ["The house is wonderful.", "I like to work in NYC."]
    model_name = "alzoubi36/pglue_piextract_priva_t5-small"
    # model_name = "alzoubi36/pglue_policy_detection_priva_t5-v1.1-small"
    # model_name = "alzoubi36/pglue_policy_detection_priva_t5-small"
    # model_name = "/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/experiments/2023-08-30/piextract/latest_model/"

    tokenizer, model = initialize_model(model_name)
    # outputs = generate_model_outputs(model, tokenizer, [task_prefix + sentence for sentence in sentences])
    # outputs = generate_model_outputs(model, tokenizer, ["Hi every one, I am Mohammad Al-Zoubi."])
    # print(outputs)
    generate_model_outputs_dataset(model, tokenizer)
