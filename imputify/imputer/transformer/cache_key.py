"""Deterministic cache key generation for training checkpoints.

Hashes model config, training args, PEFT config, schema, and seed
into a SHA-256 key so identical configurations reuse checkpoints.
"""

from peft import PeftConfig
from transformers import TrainingArguments, PreTrainedModel

import hashlib

import jsonpickle
from jsonpickle.handlers import BaseHandler, register

from imputify.imputer.transformer import SequenceFormat, JsonSequenceFormat

class _SortedSetHandler(BaseHandler):
    """Serializes sets as sorted lists for deterministic JSON output."""

    def flatten(self, obj, data): return sorted(obj)
    def restore(self, obj): return set(obj)

register(set, _SortedSetHandler, base=True)
jsonpickle.set_encoder_options('json', sort_keys=True)

IMPORTANT_FIELDS = {
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "dataloader_drop_last",

    "fp16",
    "bf16",
    "tf32",
    "fp16_full_eval",
    "bf16_full_eval",
    "fp16_opt_level",
    "half_precision_backend",
    "no_cuda",
    "use_cpu",
    "use_mps_device",

    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",

    "torch_compile",
    "torch_compile_backend",
    "torch_compile_mode",
    "torchdynamo",
    "use_liger_kernel",
    "liger_kernel_config",

    "local_rank",
    "ddp_backend",
    "ddp_find_unused_parameters",
    "ddp_bucket_cap_mb",
    "ddp_broadcast_buffers",
    "ddp_timeout",
    "fsdp",
    "fsdp_min_num_params",
    "fsdp_config",
    "fsdp_transformer_layer_cls_to_wrap",
    "deepspeed",
    "parallelism_config",
    "accelerator_config",

    "remove_unused_columns",
    "label_names",
    "length_column_name",
    "group_by_length",
    "past_index",
    "include_inputs_for_metrics",
    "include_for_metrics",

    "seed",
    "data_seed",
    "full_determinism",
    "ignore_data_skip",
}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def extract_key(
    model: PreTrainedModel,
    training_arguments: TrainingArguments,
    sequence_format: SequenceFormat,
    peft: PeftConfig | None,
    schema: dict[str, str] | None = None,
    seed: int | None = None,
) -> str:
    """Generate a deterministic cache key from training configuration.

    Args:
        model: The pretrained model (name and config are hashed).
        training_arguments: HuggingFace TrainingArguments.
        sequence_format: Row serialization format.
        peft: Optional PEFT config.
        schema: Optional column type mapping.
        seed: Optional random seed.

    Returns:
        SHA-256 hex digest identifying this configuration.
    """
    args_dict = training_arguments.to_dict()
    all_fields = {k: args_dict.get(k) for k in IMPORTANT_FIELDS}

    all_fields["sequence_format"] = sequence_format.name()

    if peft:
        peft_dict = peft.to_dict()

        if "target_modules" in peft_dict and isinstance(peft_dict["target_modules"], set):
            peft_dict["target_modules"] = sorted(peft_dict["target_modules"])

        all_fields["peft"] = peft_dict

    all_fields["model_name"] = model.name_or_path
    all_fields["model_type"] = model.__class__.__name__
    all_fields["model_config"] = model.config.to_dict()

    if schema:
        all_fields["schema"] = schema

    if seed:
        all_fields["seed"] = seed

    blob = jsonpickle.dumps(all_fields, separators=(",", ":"))
    return sha256(blob)
