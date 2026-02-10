from __future__ import annotations

import json
from typing import Protocol, Any

import numpy as np
import pandas as pd


class SequenceFormat(Protocol):
    """Protocol for row serialization formats.

    Defines how tabular rows are converted to/from text for
    the decoder-only imputer's training and inference.
    """

    def name(self) -> str: ...
    def encode(self, row: pd.Series, columns: list[str]) -> str: ...
    def decode(self, text: str, columns: list[str], column_types: dict[str, str]) -> dict[str, Any]: ...
    def create_completion_prompt(self, partial_row: pd.Series, target_field: str, columns: list[str]) -> str: ...
    def extract_completion_value(self, completed_text: str, prompt: str, target_field: str, column_types: dict[str, str]) -> Any: ...


class JsonSequenceFormat:
    """JSON key-value serialization: {"age": 25, "name": "John"}.

    Rows with missing values are serialized as open-ended JSON
    (trailing comma, no closing brace) so the model learns to
    complete partial rows.
    """

    def name(self) -> str:
        return 'json'

    def encode(self, row: pd.Series, columns: list[str]) -> str:
        """Serialize a row to JSON string, skipping NaN values.

        Args:
            row: Pandas Series representing a table row.
            columns: Column order for serialization.

        Returns:
            JSON string. Open-ended (no closing brace) if row has NaN.
        """
        json_dict = {}
        for col in columns:
            if col in row.index:
                value = row[col]
                if pd.isna(value):
                    continue

                if isinstance(value, (bool, np.bool_)):
                    json_dict[col] = bool(value)
                elif isinstance(value, (int, float, np.number)):
                    if isinstance(value, (float, np.floating)):
                        json_dict[col] = float(value)
                    else:
                        json_dict[col] = int(value)
                else:
                    json_dict[col] = str(value)

        json_str = json.dumps(json_dict, separators=(',', ':'))
        if row.isna().any():
            return json_str[:-1] + ','
        return json_str

    def decode(self, text: str, columns: list[str], column_types: dict[str, str]) -> dict[str, Any]:
        """Parse JSON text back to a column-value dict.

        Args:
            text: JSON string to parse.
            columns: Valid column names to extract.
            column_types: Mapping of column name to 'numerical'/'categorical'.

        Returns:
            Dict of parsed column values. Empty on parse failure.
        """
        result = {}
        try:
            json_data = json.loads(text.strip())
            for col_name, value in json_data.items():
                if col_name not in columns:
                    continue

                col_type = column_types.get(col_name, 'categorical')
                if col_type == 'numerical':
                    if isinstance(value, (int, float)):
                        result[col_name] = value
                    else:
                        try:
                            result[col_name] = float(value) if '.' in str(value) else int(value)
                        except (ValueError, TypeError):
                            result[col_name] = value
                else:
                    result[col_name] = value
        except json.JSONDecodeError:
            pass

        return result

    def create_completion_prompt(self, partial_row: pd.Series, target_field: str, columns: list[str]) -> str:
        """Build a prompt for the model to complete a target field.

        Places observed values as JSON context, then opens the
        target field for the model to generate its value.

        Args:
            partial_row: Row with some missing values.
            target_field: Column name to generate.
            columns: All column names.

        Returns:
            Partial JSON string ending with the target field key.
        """
        other_fields = [col for col in columns if col != target_field and col in partial_row.index and pd.notna(partial_row[col])]

        json_dict = {}
        for col in other_fields:
            value = partial_row[col]
            if isinstance(value, (bool, np.bool_)):
                json_dict[col] = bool(value)
            elif isinstance(value, (int, float, np.number)):
                json_dict[col] = float(value) if isinstance(value, (float, np.floating)) else int(value)
            else:
                json_dict[col] = str(value)

        if json_dict:
            partial_json = json.dumps(json_dict, separators=(',', ':'))[:-1]  # Remove closing }
            prompt = f'{partial_json},"{target_field}":'
        else:
            prompt = f'{{"{target_field}":'

        return prompt

    def extract_completion_value(self, completed_text: str, prompt: str, target_field: str, column_types: dict[str, str]) -> Any:
        """Extract the generated value from model output.

        Parses the completion after the prompt to extract the
        target field's value with type conversion.

        Args:
            completed_text: Full text (prompt + generation).
            prompt: The original prompt prefix.
            target_field: Column being generated.
            column_types: Mapping for type conversion.

        Returns:
            Parsed value, or None if extraction failed.
        """
        if not completed_text.startswith(prompt):
            return None

        completion_part = completed_text[len(prompt):].strip()
        if not completion_part:
            return None

        value_str = completion_part.split(',')[0].split('}')[0].strip()

        if value_str.startswith('"') and '"' in value_str[1:]:
            value_str = value_str[1:value_str.index('"', 1)]

        col_type = column_types.get(target_field, 'categorical')
        if col_type == 'numerical':
            try:
                return float(value_str) if '.' in value_str else int(value_str)
            except ValueError:
                return None

        return value_str if value_str else None
