#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pretraining the library models for T5-like span-masked language modeling on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be pretrained by this script:
https://huggingface.co/models?filter=t5
"""
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository, create_repo
from tqdm import tqdm

from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
)
from transformers.models.t5.modeling_flax_t5 import shift_tokens_right
from transformers.utils import get_full_repo_name, send_example_telemetry

from instruction_finetuning.data_preprocessing import text2text_functions
from instruction_finetuning.data_preprocessing import TASKS as PRIVACY_GLUE_TASKS
from utils import create_pdf_from_tensorboard, get_current_date, push_model_to_hub

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    training_mode: str = field(default="pretrain", metadata={"help": "Training mode: pretrain or finetune."})
    tasks: List[str] = field(default=None,
                             metadata={"help": "PrivacyGLUE tasks to train on."})

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    hub_save_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model to be pushed to the hub."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization and masking. Sequences longer than this"
                " will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            # raise ValueError("Need either a dataset name or a training/validation file.")
            pass
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            if not key == 'f1':
                summary_writer.scalar(tag, val, step - len(vals) + i + 1)
            else:
                summary_writer.scalar(tag, val[0], step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        if not metric_name == 'f1':
            summary_writer.scalar(f"eval_{metric_name}", value, step)
        else:
            summary_writer.scalar(f"eval_{metric_name}", value[0], step)


def load_arguments():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_t5_finetuning", model_args, data_args, framework="flax")
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    return data_args, model_args, training_args


class T5Evaluator:
    finetuner = None

    @classmethod
    def binary_f1(cls, y_true, y_pred):
        tp = jnp.sum((y_true == 1) & (y_pred == 1))
        fp = jnp.sum((y_true == 0) & (y_pred == 1))
        fn = jnp.sum((y_true == 1) & (y_pred == 0))
        tn = jnp.sum((y_true == 0) & (y_pred == 0))

        precision_a = tp / (tp + fp)
        recall_a = tp / (tp + fn)

        precision_b = tn / (tn + fn)
        recall_b = tn / (tn + fp)

        f1_a = 2 * (precision_a * recall_a) / (precision_a + recall_a)
        f1_b = 2 * (precision_b * recall_b) / (precision_b + recall_b)

        f1_macro = (f1_a + f1_b) / 2
        f1_micro = (tp + tn) / (tp + tn + fp + fn)

        return f1_macro, f1_micro, tp, tn, fp, fn

    @classmethod
    def opp_115(cls, finetuner, params, batch):
        ...

    @classmethod
    def policy_detection(cls, params, batch):
        finetuner = cls.finetuner
        labels = batch.pop("labels")

        logits = finetuner.model(**batch, params=params, train=False)[0]

        def _make_binary(array):
            array = array[:, :1]
            # TODO: The token id for the privacy class is 1291
            #  Check if this is persistent across all models
            #  or use tokenizer to get the id
            array = jnp.where(array == 1291, 1, 0).reshape(-1)
            return array

        # compute loss
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

        logits = _make_binary(jnp.argmax(logits, axis=-1))
        labels = _make_binary(labels)
        f1_macro, f1_micro, *other = cls.binary_f1(labels, logits)

        # summarize metrics
        metrics = {"loss": loss.mean(), "accuracy": accuracy.mean()}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        metrics["f1_macro"] = f1_macro
        metrics["f1_micro"] = f1_micro
        return metrics

    @classmethod
    def policy_ie_a(cls, finetuner, params, batch):
        ...

    @classmethod
    def piextract(cls, finetuner, params, batch):
        ...

    @classmethod
    def policy_ie_b(cls, finetuner, params, batch):
        ...

    @classmethod
    def policy_qa(cls, finetuner, params, batch):
        ...

    @classmethod
    def privacy_qa(cls, finetuner, params, batch):
        ...

    @classmethod
    def eval_functions(cls):
        return {
            "opp_115": cls.opp_115,
            "policy_detection": cls.policy_detection,
            "policy_ie_a": cls.policy_ie_a,
            "piextract": cls.piextract,
            "policy_ie_b": cls.policy_ie_b,
            "policy_qa": cls.policy_qa,
            "privacy_qa": cls.privacy_qa,
        }


class T5Finetuner:
    def __init__(self):
        self.data_args, self.model_args, self.training_args = load_arguments()
        self.logger = self.setup_logging()
        self.text2text_functions = text2text_functions
        self.eval_functions = T5Evaluator.eval_functions()

        # Set seed before initializing model.
        set_seed(self.training_args.seed)

        self.repo = self.handle_hf_repo_creation()

        # Create output directory
        self.training_args.output_dir = os.path.join(self.training_args.output_dir, get_current_date())
        self._cached_output_dir = self.training_args.output_dir
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        if not self.training_args.tasks:
            self.datasets = self.load_privacy_glue_dataset()
            self.tokenizer = self.load_tokenizer()
            self.config = self.load_model_config()

            # Tokenize our datasets.
            self.tokenized_datasets = self.tokenize_datasets()

            self.has_tensorboard, self.summary_writer = self.handle_tensorboard()
            self.model = self.load_model()
            self.linear_decay_lr_schedule_fn = None

    def finetune(self, eval_func=None, task=None):
        # Initialize our training
        rng = jax.random.PRNGKey(self.training_args.seed)
        dropout_rngs = jax.random.split(rng, jax.local_device_count())

        best_model = None
        best_eval_metric = 0.0

        # Store some constant
        num_epochs = int(self.training_args.num_train_epochs)
        train_batch_size = int(self.training_args.per_device_train_batch_size) * jax.device_count()
        per_device_eval_batch_size = int(self.training_args.per_device_eval_batch_size)
        eval_batch_size = per_device_eval_batch_size * jax.device_count()

        num_train_steps = len(self.tokenized_datasets["train"]) // train_batch_size * num_epochs

        num_of_hosts = jax.process_count()
        current_host_idx = jax.process_index()

        # Create learning rate schedule
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=self.training_args.learning_rate, transition_steps=self.training_args.warmup_steps
        )
        decay_fn = optax.linear_schedule(
            init_value=self.training_args.learning_rate,
            end_value=0,
            transition_steps=num_train_steps - self.training_args.warmup_steps,
        )
        self.linear_decay_lr_schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn], boundaries=[self.training_args.warmup_steps]
        )

        # We use Optax's "masking" functionality to not apply weight decay
        # to bias and LayerNorm scale parameters. decay_mask_fn returns a
        # mask boolean with the same structure as the parameters.
        # The mask is True for parameters that should be decayed.
        def decay_mask_fn(params):
            flat_params = traverse_util.flatten_dict(params)
            # find out all LayerNorm parameters
            layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
            layer_norm_named_params = {
                layer[-2:]
                for layer_norm_name in layer_norm_candidates
                for layer in flat_params.keys()
                if layer_norm_name in "".join(layer).lower()
            }
            flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in
                         flat_params}
            return traverse_util.unflatten_dict(flat_mask)

        # create adam optimizer
        if self.training_args.adafactor:
            # We use the default parameters here to initialize adafactor, For more details about the parameters
            # please check https://github.com/deepmind/optax/blob/ed02befef9bf81cbbf236be3d2b0e032e9ed4a40/optax/_src
            # /alias.py#L74
            optimizer = optax.adafactor(
                learning_rate=self.linear_decay_lr_schedule_fn,
            )
        else:
            optimizer = optax.adamw(
                learning_rate=self.linear_decay_lr_schedule_fn,
                b1=self.training_args.adam_beta1,
                b2=self.training_args.adam_beta2,
                weight_decay=self.training_args.weight_decay,
                mask=decay_mask_fn,
            )

        # Setup train state
        state = train_state.TrainState.create(apply_fn=self.model.__call__, params=self.model.params, tx=optimizer)

        # Define gradient update step fn

        # Create parallel version of the train step
        p_train_step = jax.pmap(self.train_step, "batch", donate_argnums=(0,))

        if not eval_func:
            p_eval_step = jax.pmap(self.eval_step, "batch", donate_argnums=(0,))
        else:
            p_eval_step = jax.pmap(eval_func, "batch", donate_argnums=(0,))

        # Replicate the train state on each device
        state = jax_utils.replicate(state)

        # Count steps with no improvement
        no_improvement_count = 0

        train_time = 0
        epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
        for epoch in epochs:
            # ======================== Training ================================

            if no_improvement_count >= 40:
                break

            train_start = time.time()
            train_metrics = []

            # Create sampling rng
            rng, input_rng = jax.random.split(rng)

            # Generate an epoch by shuffling sampling indices from the train dataset
            num_train_samples = len(self.tokenized_datasets["train"])

            # Avoid using jax.numpy here in case of TPU training
            train_samples_idx = np.random.permutation(np.arange(num_train_samples))
            train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

            # Gather the indexes for creating the batch and do a training step
            for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):
                model_inputs = {
                    'input_ids': np.asarray(
                        [self.tokenized_datasets["train"][int(idx)]['input_ids'] for idx in batch_idx]),
                    'labels': np.asarray([self.tokenized_datasets["train"][int(idx)]['labels'] for idx in batch_idx])}
                model_inputs["decoder_input_ids"] = shift_tokens_right(
                    model_inputs["labels"], self.model.config.pad_token_id, self.model.config.decoder_start_token_id
                )

                local_host_model_inputs = {
                    key: np.split(model_inputs[key], num_of_hosts, axis=0)[current_host_idx]
                    for key, value in model_inputs.items()
                }

                # Model forward
                model_inputs = shard(local_host_model_inputs)
                state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
                train_metrics.append(train_metric)

                cur_step = epoch * (num_train_samples // train_batch_size) + step

                if cur_step % self.training_args.logging_steps == 0 and cur_step > 0:
                    # Save metrics
                    train_metric = jax_utils.unreplicate(train_metric)
                    train_time += time.time() - train_start
                    if self.has_tensorboard and jax.process_index() == 0:
                        write_train_metric(self.summary_writer, train_metrics, train_time, cur_step)

                    epochs.write(
                        f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate:"
                        f" {train_metric['learning_rate'].mean()})"
                    )

                    train_metrics = []

                if cur_step % self.training_args.eval_steps == 0 and cur_step > 0:
                    # ======================== Evaluating ==============================
                    num_eval_samples = len(self.tokenized_datasets["validation"])
                    # Avoid using jax.numpy here in case of TPU training
                    eval_samples_idx = np.arange(num_eval_samples)
                    eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size, drop_last=False)

                    eval_metrics = []
                    for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
                        model_inputs = {
                            'input_ids': np.asarray(
                                [self.tokenized_datasets["validation"][int(idx)]['input_ids'] for idx in batch_idx]),
                            'labels': np.asarray(
                                [self.tokenized_datasets["validation"][int(idx)]['labels'] for idx in batch_idx])}
                        model_inputs["decoder_input_ids"] = shift_tokens_right(
                            model_inputs["labels"], self.model.config.pad_token_id,
                            self.model.config.decoder_start_token_id
                        )

                        # Model forward
                        if eval_func:
                            T5Evaluator.finetuner = self
                        metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                            state.params, model_inputs, min_device_batch=per_device_eval_batch_size
                        )

                        eval_metrics.append(metrics)

                    # get eval metrics
                    eval_metrics = get_metrics(eval_metrics)
                    eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

                    if eval_metrics['accuracy'] > best_eval_metric:
                        best_eval_metric = eval_metrics['accuracy']
                        self.save_model(cur_step, state, os.path.join(self.training_args.output_dir, 'best_model'))
                        self.logger.info(f"Saving best model...")
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    # Update progress bar
                    epochs.write(
                        f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']}")
                    # f"F1_macro: {eval_metrics['f1_macro']}, F1_micro: {eval_metrics['f1_micro']})")

                    # Save metrics
                    if self.has_tensorboard and jax.process_index() == 0:
                        write_eval_metric(self.summary_writer, eval_metrics, cur_step)

                if cur_step % self.training_args.save_steps == 0 and cur_step > 0:
                    self.save_model(cur_step, state, os.path.join(self.training_args.output_dir, 'latest_model'))

        self.create_training_report()

        # ======================== Evaluation ================================
        # Eval after training
        if self.training_args.do_eval:
            eval_metrics = self.evaluate(eval_batch_size, p_eval_step, per_device_eval_batch_size, state)

            if jax.process_index() == 0:
                eval_metrics = {f"eval_{metric_name}": value for metric_name, value in eval_metrics.items()}
                path = os.path.join(self.training_args.output_dir, "eval_results.json")
                with open(path, "w") as f:
                    json.dump(eval_metrics, f, indent=4, sort_keys=True)

    def save_model(self, cur_step, state, output_dir):
        # save checkpoint after each epoch and push checkpoint to the hub
        if jax.process_index() == 0:
            params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
            self.model.save_pretrained(output_dir, params=params)
            self.tokenizer.save_pretrained(output_dir)
            if self.training_args.push_to_hub:
                self.repo.push_to_hub(commit_message=f"Saving weights and logs of step {cur_step}",
                                      blocking=False)

    def evaluate(self, eval_batch_size, p_eval_step, per_device_eval_batch_size, state):
        num_eval_samples = len(self.tokenized_datasets["validation"])
        # Avoid using jax.numpy here in case of TPU training
        eval_samples_idx = np.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size, drop_last=False)
        eval_metrics = []
        for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=2)):
            model_inputs = {
                'input_ids': np.asarray(
                    [self.tokenized_datasets["validation"][int(idx)]['input_ids'] for idx in batch_idx]),
                'labels': np.asarray(
                    [self.tokenized_datasets["validation"][int(idx)]['labels'] for idx in batch_idx])}
            model_inputs["decoder_input_ids"] = shift_tokens_right(
                model_inputs["labels"], self.model.config.pad_token_id, self.model.config.decoder_start_token_id
            )

            # Model forward
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params, model_inputs, min_device_batch=per_device_eval_batch_size
            )
            eval_metrics.append(metrics)
        # get eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(lambda metric: jnp.mean(metric).item(), eval_metrics)
        return eval_metrics

    def load_model(self):
        if self.model_args.model_name_or_path:
            model = FlaxT5ForConditionalGeneration.from_pretrained(
                self.model_args.model_name_or_path,
                config=self.model_args.model_name_or_path,
                seed=self.training_args.seed,
                dtype=getattr(jnp, self.model_args.dtype),
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        else:
            self.config.vocab_size = len(self.tokenizer)
            model = FlaxT5ForConditionalGeneration(
                self.config,
                seed=self.training_args.seed,
                dtype=getattr(jnp, self.model_args.dtype),
            )
        return model

    def handle_tensorboard(self):
        # Enable tensorboard only on the master node
        has_tensorboard = is_tensorboard_available()
        if has_tensorboard and jax.process_index() == 0:
            try:
                from flax.metrics.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=Path(self.training_args.output_dir))
            except ImportError as ie:
                has_tensorboard = False
                self.logger.warning(
                    f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
                )
        else:
            self.logger.warning(
                "Unable to display metrics through TensorBoard because the package is not installed: "
                "Please run pip install tensorboard to enable."
            )
        return has_tensorboard, summary_writer

    def tokenize_datasets(self):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        if self.training_args.do_train:
            column_names = self.datasets["train"].column_names
        else:
            column_names = self.datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # Since we make sure that all sequences are of the same length, no attention_mask is needed.
        def tokenize_function(examples):
            data = self.tokenizer(examples[text_column_name],
                                  max_length=max_seq_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_attention_mask=False)
            data["labels"] = self.tokenizer(examples['label'],
                                            max_length=max_seq_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_attention_mask=False)["input_ids"]
            return data

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        tokenized_datasets = self.datasets.map(
            tokenize_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        return tokenized_datasets

    def load_model_config(self):
        if self.model_args.config_name:
            config = T5Config.from_pretrained(
                self.model_args.config_name,
                cache_dir=self.model_args.cache_dir,
                vocab_size=len(self.tokenizer),
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        elif self.model_args.model_name_or_path:
            config = T5Config.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        else:
            config = CONFIG_MAPPING[self.model_args.model_type]()
            self.logger.warning("You are instantiating a new config instance from scratch.")
        return config

    def load_tokenizer(self):
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame,
        # etc) at https://huggingface.co/docs/datasets/loading_datasets.html. Load pretrained model and tokenizer
        if self.model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.tokenizer_name,
                cache_dir=self.model_args.cache_dir,
                use_fast=self.model_args.use_fast_tokenizer,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        elif self.model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                use_fast=self.model_args.use_fast_tokenizer,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        return tokenizer

    def load_finetuning_dataset(self, task='policy_detection'):
        # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below) or
        # just provide the name of one of the public datasets available on the hub at
        # https://huggingface.co/datasets/ (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found. You can easily tweak this behavior (see below).
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            if "validation" not in datasets.keys():
                datasets["validation"] = load_dataset(
                    self.data_args.dataset_name,
                    self.data_args.dataset_config_name,
                    split=f"train[:{self.data_args.validation_split_percentage}%]",
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
                datasets["train"] = load_dataset(
                    self.data_args.dataset_name,
                    self.data_args.dataset_config_name,
                    split=f"train[{self.data_args.validation_split_percentage}%:]",
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=self.model_args.cache_dir,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            if "validation" not in datasets.keys():
                datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.data_args.validation_split_percentage}%]",
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
                datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.data_args.validation_split_percentage}%:]",
                    cache_dir=self.model_args.cache_dir,
                    use_auth_token=True if self.model_args.use_auth_token else None,
                )
        return datasets

    def load_privacy_glue_dataset(self, task='policy_detection'):
        if task not in self.text2text_functions['privacy_glue'].keys():
            raise ValueError(f"Task {task} not found in PrivacyGLUE benchmark.")
        return self.text2text_functions['privacy_glue'][task]()

    def handle_hf_repo_creation(self):
        # Handle the repository creation
        if self.training_args.push_to_hub:
            if self.training_args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(self.training_args.output_dir).absolute().name, token=self.training_args.hub_token
                )
            else:
                repo_name = self.training_args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=self.training_args.hub_token)
            repo = Repository(self.training_args.output_dir, clone_from=repo_name, token=self.training_args.hub_token)
            return repo

    def setup_logging(self):
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            level=logging.INFO,
            datefmt="[%X]",
        )
        # Log on each process the small summary:
        logger = logging.getLogger(__name__)
        # Set the verbosity to info of the Transformers logger (on main process only):
        logger.info(f"Training/evaluation parameters {self.training_args}")
        return logger

    def train_step(self, state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            labels = batch.pop("labels")

            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]

            # compute loss
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()

            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": self.linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    def eval_step(self, params, batch):
        labels = batch.pop("labels")

        logits = self.model(**batch, params=params, train=False)[0]

        # compute loss
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels)

        # summarize metrics
        metrics = {"loss": loss.mean(), "accuracy": accuracy.mean()}
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    def create_training_report(self):
        self.logger.info('Generating training report...')
        create_pdf_from_tensorboard(self.training_args.output_dir, self.training_args.output_dir)

    def finetune_on_privacy_glue(self):
        tasks = self.training_args.tasks
        self.logger.info(f'======================= Finetuning {self.model_args.model_name_or_path} on the following '
                         f'tasks: =======================')
        self.logger.info(f'{tasks}')

        for task in tasks:
            if task not in PRIVACY_GLUE_TASKS:
                raise ValueError(f"Task \"{task}\" not found in PrivacyGLUE benchmark. "
                                 f"Task must be one of {PRIVACY_GLUE_TASKS}")

            # Create a task directory
            self.training_args.output_dir = os.path.join(self._cached_output_dir, task)
            Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

            self.datasets = self.load_privacy_glue_dataset(task=task)
            self.tokenizer = self.load_tokenizer()
            self.config = self.load_model_config()

            # Tokenize our datasets.
            self.tokenized_datasets = self.tokenize_datasets()

            self.has_tensorboard, self.summary_writer = self.handle_tensorboard()
            self.model = self.load_model()
            self.linear_decay_lr_schedule_fn = None

            self.logger.info(f'======================= Finetuning on task {task}... =======================')

            self.finetune(task=task)

            if self.model_args.hub_save_name_or_path:
                # Free memory
                self.model = None
                try:
                    push_model_to_hub(os.path.join(self.training_args.output_dir, 'best_model'),
                                      f'pglue_{task}_{self.model_args.hub_save_name_or_path}')
                except Exception as e:
                    self.logger.info(f'Error pushing best model to hub: {e}')
                    push_model_to_hub(os.path.join(self.training_args.output_dir, 'latest_model'),
                                      f'pglue_{task}_{self.model_args.hub_save_name_or_path}')


if __name__ == "__main__":
    finetuner = T5Finetuner()
    finetuner.finetune_on_privacy_glue()
