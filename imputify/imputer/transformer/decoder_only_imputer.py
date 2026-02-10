from __future__ import annotations

import random
import threading
from typing import Literal, Any, Iterator

import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModelForCausalLM
from sklearn.base import TransformerMixin, BaseEstimator

from transformers import (
    TrainingArguments,
    AutoTokenizer,
    Trainer,
    PrinterCallback,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from .cache_key import extract_key
from .format import SequenceFormat, JsonSequenceFormat
from .dataset import SimpleTextDataset, ImputerDataCollator
from .pushable_iterator import PushableIterator, PushableTrainingCallback, TrainingStep, ImputationStep
from imputify.shared import seed_everything, global_seed, cleanup_gpu
from imputify.imputer.mixins import BaseImputerMixin


class DebugTrainer(Trainer):
    """Trainer that randomly prints batch samples for debugging.

    Args:
        debug_sample_ratio: Probability of printing a batch sample per step.
        seed: Random seed for reproducible sampling.
    """

    def __init__(self, *args, debug_sample_ratio: float = 0.5, seed: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_sample_ratio = debug_sample_ratio
        self._rng = random.Random(seed)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._rng.random() < self.debug_sample_ratio:
            input_ids = inputs["input_ids"][0]
            text = self.data_collator.tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"\n[DEBUG] Step {self.state.global_step} - Batch sample: {text}\n")

        return super().training_step(model, inputs, num_items_in_batch)


class DecoderOnlyImputer(BaseImputerMixin, TransformerMixin, BaseEstimator):
    """Fine-tuned decoder-only LLM for tabular data imputation.

    Based on the GReaT approach: each row is serialized as text (JSON
    key-value pairs), then a causal language model is fine-tuned with
    standard next-token prediction (cross-entropy loss).

    The training objective maximizes:

        p(t) = Π p(w_k | w_1, ..., w_{k-1})

    over all serialized rows. Column order is randomly permuted per
    sample to prevent positional bias and learn an order-invariant
    joint distribution over features.

    At inference, observed values are placed as a prompt prefix and
    the model generates missing values via conditional sampling:

        p(V_missing | V_observed = v_observed)

    with temperature-scaled sampling and a retry mechanism that
    escalates temperature/top_k on parse failures.

    Supports PEFT (LoRA) and quantization (4-bit/8-bit) for
    efficient fine-tuning of large models.

    Reference: Borisov et al., 2023 - "Language Models are Realistic
    Tabular Data Generators" (GReaT)

    Parameters
    ----------
    model : PreTrainedModel
        Base HuggingFace causal LM (e.g. GPT-2, Qwen).
    peft : PeftConfig, optional
        PEFT config (e.g. LoraConfig) for parameter-efficient fine-tuning.
    tokenizer : PreTrainedTokenizerBase, optional
        Tokenizer. Inferred from model if not provided.
    sequence_format : SequenceFormat or 'json', default='json'
        Row serialization format.
    training_args : TrainingArguments, optional
        HuggingFace training arguments.
    temperature : float, default=0.2
        Sampling temperature for generation.
    top_p : float, default=0.95
        Nucleus sampling threshold.
    top_k : int, default=50
        Top-k sampling. 0 disables.
    max_new_tokens : int, default=50
        Max tokens to generate per completion.
    resume_from_checkpoint : bool, default=True
        Resume training from cached checkpoint if available.
    debug_samples : float, optional
        If set, use DebugTrainer with this sample ratio.
    validation_split : float, optional
        Fraction of data to use for validation.
    seed : int, optional
        Random seed. Falls back to global seed.
    max_retries : int, default=3
        Max retry attempts per cell on parse failure.
    retry_temperatures : list[float], optional
        Escalating temperatures for retries.
    retry_top_ks : list[int], optional
        Escalating top_k values for retries.
    early_stopping_patience : int | None, default=None
        Stop training after this many evaluations without improvement
        in eval_loss. Requires validation_split to be set. None disables.
    early_stopping_threshold : float, default=0.0
        Minimum improvement in eval_loss to count as progress.
    inference_batch_size : int, default=64
        Batch size for generation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft: PeftConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        sequence_format: SequenceFormat | Literal['json'] = 'json',
        training_args: TrainingArguments | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 50,
        resume_from_checkpoint: bool = True,
        debug_samples: float | None = None,
        validation_split: float | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_threshold: float = 0.0,
        seed: int | None = None,
        max_retries: int = 3,
        retry_temperatures: list[float] | None = None,
        retry_top_ks: list[int] | None = None,
        inference_batch_size: int = 64,
    ):
        model.enable_input_require_grads()

        if peft is not None:
            is_quantized = getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False)
            if is_quantized:
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
            model = PeftModelForCausalLM(model, peft)

        self.training_args = training_args or TrainingArguments(
            output_dir='./imputify',
            num_train_epochs=10,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            logging_steps=10,
        )

        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=self.training_args.gradient_checkpointing_kwargs
        )

        self.model = model
        self.tokenizer = tokenizer or tokenizer_from_model(model)
        self.format = JsonSequenceFormat() if sequence_format == 'json' else sequence_format
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.resume_from_checkpoint = resume_from_checkpoint
        self.debug_samples = debug_samples
        self.validation_split = validation_split
        self.peft = peft
        self.seed = seed or global_seed()
        self.max_retries = max_retries
        self.retry_temperatures = retry_temperatures or [0.4, 0.7, 1.0]
        self.retry_top_ks = retry_top_ks or [100, 200, 0]
        self.inference_batch_size = inference_batch_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.inference_batch_size <= 0:
            raise ValueError(f"inference_batch_size must be > 0, got {self.inference_batch_size}")

        self._fitted = False
        self._columns: list[str] = []
        self._column_types: dict[str, str] = {}
        self._max_length: int = 128

        self._base_output_dir = self.training_args.output_dir

    @property
    def _cache_key(self) -> str:
        if not self._column_types:
            raise RuntimeError("Cannot compute cache key before fitting.")
        return extract_key(
            self.model,
            self.training_args,
            self.format,
            self.peft,
            schema=self._column_types,
            seed=self.seed,
        )

    def streamed_fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Any | None = None
    ) -> Iterator[TrainingStep]:
        """Fine-tune the model on data with missing values.

        Runs training in a background thread and yields progress steps.

        Args:
            X: Input data containing NaN values.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            Iterator of TrainingStep with loss and progress info.
        """
        seed_everything(self.seed)

        X_df = self._validate_fit_input(X)
        self._setup_columns(X_df)

        self.training_args.output_dir = f'{self._base_output_dir}/{self._cache_key}'

        self._max_length = self._infer_max_length(X_df)

        X_df = X_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        train_df = X_df
        eval_dataset = None

        if self.validation_split is not None:
            n_samples = len(X_df)
            n_val = int(n_samples * self.validation_split)
            rng = np.random.RandomState(self.seed)
            indices = rng.permutation(n_samples)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            train_df = X_df.iloc[train_indices].reset_index(drop=True)
            val_df = X_df.iloc[val_indices].reset_index(drop=True)
            eval_dataset = SimpleTextDataset(val_df, self.format, self.tokenizer, self._max_length, seed=self.seed)

        dataset = SimpleTextDataset(train_df, self.format, self.tokenizer, self._max_length, seed=self.seed)

        data_collator = ImputerDataCollator(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self._max_length,
        )

        stream = PushableIterator()

        callbacks = [PushableTrainingCallback(stream)]

        if self.early_stopping_patience is not None and eval_dataset is not None:
            self.training_args.eval_strategy = "epoch"
            self.training_args.save_strategy = "epoch"
            self.training_args.load_best_model_at_end = True
            self.training_args.metric_for_best_model = "eval_loss"
            self.training_args.greater_is_better = False
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=self.early_stopping_threshold,
            ))

        TrainerClass = DebugTrainer if self.debug_samples else Trainer
        trainer_kwargs = dict(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        if self.debug_samples:
            trainer_kwargs["debug_sample_ratio"] = self.debug_samples
            trainer_kwargs["seed"] = self.seed

        trainer = TrainerClass(**trainer_kwargs)
        trainer.remove_callback(PrinterCallback)

        thread = threading.Thread(
            target=self._do_train,
            args=(trainer,stream,),
            daemon=True,
        )

        thread.start()

        return stream

    def _do_train(self, trainer: Trainer, stream: PushableIterator):
        try:
            try:
                trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
            except Exception as e:
                if "No valid checkpoint found" not in str(e):
                    raise

                trainer.train()

            self._fitted = True
        finally:
            stream.close()

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any | None = None) -> DecoderOnlyImputer:
        """Fine-tune the model on data with missing values.

        Blocks until training completes (consumes streamed_fit).

        Args:
            X: Input data containing NaN values.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            The fitted imputer.
        """
        for _ in self.streamed_fit(X, y):
            pass

        return self

    def streamed_transform(
        self,
        X: np.ndarray | pd.DataFrame
    ) -> Iterator[ImputationStep | np.ndarray | pd.DataFrame]:
        """Impute missing values with progress streaming per cell.

        Generates completions for each missing cell, retrying with
        escalating temperature/top_k on parse failures.

        Args:
            X: Data with missing values to impute.

        Yields:
            ImputationStep with progress after each cell attempt.
            Final item is the imputed DataFrame/array.
        """
        X_df, return_numpy = self._validate_transform_input(X)

        if not X_df.isnull().any().any():
            yield X_df.values if return_numpy else X_df
            return

        X_imputed = X_df.copy()

        mask = X_df.isnull()
        rows_total = mask.any(axis=1).sum()
        cells_total = mask.sum().sum()
        cells_imputed = 0

        attempts = {}
        exhausted = set()
        total_retries = 0
        total_failures = 0
        max_iterations = cells_total * (self.max_retries + 2)
        iteration = 0

        while X_imputed.isnull().any().any():
            iteration += 1
            if iteration > max_iterations:
                break
            row_info = []

            def prompts_with_info():
                for idx, row in X_imputed.iterrows():
                    missing_cols = row.index[row.isnull()].tolist()
                    if not missing_cols:
                        continue

                    target_col = None
                    for col in missing_cols:
                        if (idx, col) not in exhausted:
                            target_col = col
                            break

                    if target_col is None:
                        continue

                    key = (idx, target_col)
                    prompt = self.format.create_completion_prompt(row, target_col, self._columns)
                    row_info.append((idx, target_col, prompt, key))
                    yield prompt

            prompts_list = list(prompts_with_info())
            if not prompts_list:
                break

            max_attempt = max((attempts.get(info[3], 0) for info in row_info), default=0)
            if max_attempt > 0:
                temp, top_k = self._get_retry_params(max_attempt - 1)
            else:
                temp, top_k = None, None

            completions = self._complete_batch(prompts_list, temp, top_k)

            for i, completion in enumerate(completions):
                idx, target_col, prompt, key = row_info[i]

                value = self.format.extract_completion_value(
                    completion,
                    prompt,
                    target_col,
                    self._column_types,
                )

                if value is not None:
                    X_imputed.at[idx, target_col] = value
                    cells_imputed += 1
                    attempts.pop(key, None)
                else:
                    attempts[key] = attempts.get(key, 0) + 1
                    total_retries += 1
                    if attempts[key] > self.max_retries:
                        exhausted.add(key)
                        total_failures += 1

                rows_completed = rows_total - X_imputed.isnull().any(axis=1).sum()

                yield ImputationStep(
                    rows_completed=rows_completed,
                    rows_total=rows_total,
                    cells_imputed=cells_imputed,
                    cells_total=cells_total,
                    retries=total_retries,
                    failures=total_failures,
                )

        yield X_imputed.values if return_numpy else X_imputed

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Impute missing values using the fine-tuned model.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        result = None
        for item in self.streamed_transform(X):
            if not isinstance(item, ImputationStep):
                result = item
        if result is None:
            raise RuntimeError("transform() produced no result — stream yielded only ImputationStep items")
        return result

    def _infer_max_length(self, df: pd.DataFrame) -> int:
        """Calculate max token length across all rows for padding."""
        columns = list(df.columns)
        return max(
            len(self.tokenizer.encode(self.format.encode(row, columns)))
            for _, row in df.iterrows()
        )

    def _get_retry_params(self, attempt: int) -> tuple[float, int]:
        """Get escalating (temperature, top_k) for a retry attempt."""
        temp_idx = min(attempt, len(self.retry_temperatures) - 1)
        top_k_idx = min(attempt, len(self.retry_top_ks) - 1)
        return self.retry_temperatures[temp_idx], self.retry_top_ks[top_k_idx]

    def _complete_batch(
        self,
        prompts: list[str],
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> list[str]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: List of partial JSON strings to complete.
            temperature: Override temperature (used for retries).
            top_k: Override top_k (used for retries).

        Returns:
            List of full text completions (prompt + generated).
        """
        temp = temperature if temperature is not None else self.temperature
        tk = top_k if top_k is not None else self.top_k

        self.model.eval()
        all_completions = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for i in range(0, len(prompts), self.inference_batch_size):
                batch_prompts = prompts[i:i + self.inference_batch_size]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                ).to(device)

                input_lengths = inputs['attention_mask'].sum(dim=1)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=temp,
                    top_p=self.top_p,
                    top_k=tk if tk > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for j, output in enumerate(outputs):
                    input_len = input_lengths[j].item()
                    generated_tokens = output[int(input_len):]
                    generated_text = self.tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )
                    all_completions.append(batch_prompts[j] + generated_text)

                del inputs, outputs
                cleanup_gpu()

        return all_completions

def tokenizer_from_model(model: PreTrainedModel) -> AutoTokenizer:
    """Create a left-padded tokenizer from a model's config.

    Uses the model's name_or_path to load the matching tokenizer.
    Sets pad_token to eos_token if not defined.

    Args:
        model: HuggingFace PreTrainedModel.

    Returns:
        Configured AutoTokenizer with left padding.
    """
    model_name = getattr(model.config, 'name_or_path', model.__class__.__name__)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    return tokenizer
