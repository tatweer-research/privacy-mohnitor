import json
import logging

from instruction_finetuning.models_evaluation import (
    list_models,
    huggingface_search_params,
    initialize_model,
    generate_model_outputs_dataset,
    generate_and_evaluate,
    get_base_model_name,
    get_task_name,
    TAKS_EVALUATION_FUNCTIONS,
)

from pathlib import Path
import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GeneralConfig(BaseModel):
    remote_user: str
    path_to_results_json: str
    path_to_model_outputs: str
    batch_sizes: dict
    requested_sizes: list
    requested_base_models: list
    requested_tasks: list
    examples_limit: int


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
        logger.info(f"Current configuration:\n{json.dumps(self.general_config.__dict__, indent=4)}")
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
            logger.info("Hugging Face cache cleared successfully.")
        except subprocess.CalledProcessError as e:
            logger.info(f"Error clearing Hugging Face cache: {e}")

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
        model_names = list(
            filter(lambda x: get_base_model_name(x) in self.general_config.requested_base_models, model_names))
        model_names = list(filter(lambda x: get_task_name(x) in self.general_config.requested_tasks, model_names))
        return model_names

    def evaluate_model(self, model_name, task):
        logger.info(f"Running inference on model {model_name} with task {task}...")

        model_save_name = model_name

        if get_task_name(model_name) == 'multitask':
            model_save_name = model_name.replace('multitask', task)

        # Free space for new models
        if get_task_name(model_name) != 'multitask':
            self.remove_huggingface_cache()

        tokenizer_name = self.get_tokenizer_name(model_name)
        max_generation_length = 5 if task == 'privacy_qa' else 512
        logger.info(f"Limiting generation length on model {model_name} to {max_generation_length} tokens...")
        try:
            model_outputs_dir = Path(self.general_config.path_to_model_outputs).joinpath(model_save_name)
            model_outputs_json = model_outputs_dir / 'outputs.json'
            if model_outputs_json.exists():
                logger.info(f"Model {model_name} already evaluated. Skipping...")
                return None

            model_outputs_dir.mkdir(parents=True, exist_ok=True)
            evaluation_result = generate_and_evaluate(model_name=model_name,
                                                      batch_size=self.get_batch_size(model_name),
                                                      examples_limit=self.general_config.examples_limit,
                                                      tokenizer_name=self.get_tokenizer_name(model_name),
                                                      pglue_task=task,
                                                      split='test',
                                                      max_generation_length=max_generation_length,
                                                      output_json=model_outputs_json)

            self.results[model_save_name] = evaluation_result
            with open(self.general_config.path_to_results_json, 'w', encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.info(f"Error running inference on model {model_name}:\n{e}")
            logger.info(f"Batch size: {self.get_batch_size(model_name)}")
            logger.info(f"Tokenizer name: {tokenizer_name}")
            logger.info(f"Task: {get_task_name(model_name)}")
            logger.info("Skipping model...")

    def run(self):
        available_models = list_models(huggingface_search_params)
        model_names = [model['modelId'] for model in available_models]
        model_names = self.filter_models(model_names)

        logger.info(f"Found {len(model_names)} models...")
        for model_name in model_names:

            task = get_task_name(model_name)
            if task != 'multitask':
                self.evaluate_model(model_name, task)
            else:
                for task in self.general_config.requested_tasks:
                    self.evaluate_model(model_name, task)


if __name__ == '__main__':
    config = parse_config(
        "/home/Mohammad.Al-Zoubi/privacy-mohnitor/instruction_finetuning/models_evaluation/config.yaml")
    pipeline = T5EvaluationPipeline(config)
    pipeline.run()
