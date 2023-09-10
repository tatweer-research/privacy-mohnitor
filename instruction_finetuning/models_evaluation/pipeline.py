import json

from instruction_finetuning.models_evaluation import (
    list_models,
    huggingface_search_params,
    initialize_model,
    generate_model_outputs_dataset,
    get_base_model_name,
    get_task_name,
    TAKS_EVALUATION_FUNCTIONS
)

from pathlib import Path
import yaml
from pydantic import BaseModel


class GeneralConfig(BaseModel):
    remote_user: str
    path_to_results_json: str
    path_to_model_outputs: str
    batch_sizes: dict
    requested_sizes: list


class AppConfig(BaseModel):
    general: GeneralConfig


def parse_config(yaml_config):
    # Parse the YAML
    with open(yaml_config, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)

    # Create a Pydantic object
    app_config = AppConfig(**config_dict)
    return app_config


class T5EvaluationPipeline:
    def __init__(self, config):
        self.general_config = config.general
        self.results = {}

    @staticmethod
    def remove_huggingface_cache():
        """Remove huggingface cache in /home/Username/.cache/huggingface/hub/ on the remote machine to free up space
        for new models."""
        import subprocess

        try:
            # Define the shell command to remove the cache directory
            command = "sudo rm -rf ~/.cache/huggingface/hub/"

            # Execute the command using subprocess
            subprocess.run(command, shell=True, check=True)
            print("Hugging Face cache cleared successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error clearing Hugging Face cache: {e}")

    @staticmethod
    def pglue_model(model_name):
        if model_name.split('/')[-1].startswith("pglue_"):
            return True
        return False

    def get_batch_size(self, model_name):
        model_size = model_name.split('-')[-1]
        return self.general_config.batch_sizes[model_size]

    @staticmethod
    def get_tokenizer_name(model_name):
        base_model_name = get_base_model_name(model_name)
        tokenizer_name = base_model_name.split('_')[-1] + '-' + model_name.split('-')[-1]
        if 'v1.1' in tokenizer_name:
            tokenizer_name = 'google' + '/' + tokenizer_name.replace('.', '_')
        return tokenizer_name

    def filter_models(self, model_names):
        model_names = list(filter(self.pglue_model, model_names))
        model_names = list(filter(lambda x: x.split('-')[-1] in self.general_config.requested_sizes, model_names))
        return model_names

    def run(self):
        available_models = list_models(huggingface_search_params)
        model_names = [model['modelId'] for model in available_models]
        model_names = self.filter_models(model_names)

        print(f"Found {len(model_names)} models...")
        for model_name in model_names:
            print(f"Running inference on model {model_name}...")

            # Free space for new models
            self.remove_huggingface_cache()
            tokenizer_name = self.get_tokenizer_name(model_name)
            max_generation_length = 5 if get_task_name(model_name) == 'privacy_qa' else 512
            print(f"Limiting generation length on model {model_name} to {max_generation_length} tokens...")

            try:
                model_outputs_dir = Path(self.general_config.path_to_model_outputs).joinpath(model_name)
                model_outputs_json = model_outputs_dir / 'outputs.json'
                if model_outputs_json.exists():
                    print(f"Model {model_name} already evaluated. Skipping...")
                    continue
                tokenizer, model = initialize_model(model_name, tokenizer_name=tokenizer_name)
                model_outputs_dir.mkdir(parents=True, exist_ok=True)

                generate_model_outputs_dataset(model,
                                               tokenizer,
                                               model_outputs_json,
                                               pglue_task=get_task_name(model_name),
                                               batch_size=self.get_batch_size(model_name),
                                               max_generation_length=max_generation_length)
                task = get_task_name(model_name)
                evaluation_result = TAKS_EVALUATION_FUNCTIONS[task](model_outputs_json)
                self.results[model_name] = evaluation_result
                with open(self.general_config.path_to_results_json, 'w', encoding="utf-8") as f:
                    json.dump(self.results, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"Error running inference on model {model_name}: {e}")
                print(f"Batch size: {self.get_batch_size(model_name)}")
                print(f"Tokenizer name: {tokenizer_name}")
                print(f"Task: {get_task_name(model_name)}")
                print("Skipping model...")
                continue


if __name__ == '__main__':
    config = parse_config(
        "/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/models_evaluation/config.yaml")
    pipeline = T5EvaluationPipeline(config)
    pipeline.run()
