from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.pipelines.base import Dataset

from .format import SequenceFormat

class SimpleTextDataset(Dataset):
    """Training dataset that serializes rows with random column shuffling.

    Each __getitem__ call applies a fresh column permutation to
    prevent the model from learning positional dependencies.
    """

    def __init__(self, dataframe: pd.DataFrame, sequence_format: SequenceFormat, tokenizer: PreTrainedTokenizerBase, max_length: int, seed: int | None = None):
        self.dataframe = dataframe.copy()
        self.sequence_format = sequence_format
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = list(dataframe.columns)
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataframe.iloc[idx]

        shuffled_columns = self.columns.copy()
        self._rng.shuffle(shuffled_columns)

        text = self.sequence_format.encode(row, shuffled_columns)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class ImputerDataCollator(DataCollatorWithPadding):
    """Collator that creates causal LM labels from input_ids.

    Pads sequences and sets labels = input_ids, masking padding
    positions with -100 so they're ignored in the loss.
    """

    def __call__(self, features: list[dict[str, Any]]):
        padding = self.padding if self.padding != True else 'longest'
        batch = self.tokenizer.pad(
            features,
            padding=padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["attention_mask"] == 0] = -100
        return batch

