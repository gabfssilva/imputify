"""Experiment configuration: YAML parsing and dynamic object instantiation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from imputify.missing import PatternConfig


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    dataset: str
    patterns: tuple[PatternConfig, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    imputer_name: str
    imputer_cfg: dict[str, Any]
    scenario: ScenarioConfig
    rate: float
    seed: int


def load_experiments(path: Path) -> dict[str, Any]:
    """Load experiments configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_scenario(scenario_dict: dict[str, Any]) -> ScenarioConfig:
    """Parse scenario from YAML dict."""
    patterns = tuple(
        PatternConfig(
            incomplete_vars=p["incomplete_vars"],
            mechanism=p["mechanism"],
        )
        for p in scenario_dict["patterns"]
    )
    return ScenarioConfig(
        name=scenario_dict["name"],
        dataset=scenario_dict["dataset"],
        patterns=patterns,
    )


def build_config(
    experiments: dict[str, Any],
    imputer: str,
    scenario: str,
    rate: float,
    seed: int,
) -> ExperimentConfig:
    """Build experiment config from CLI arguments."""
    imputer_cfg = experiments["imputers"][imputer]
    scenario_dict = next(s for s in experiments["scenarios"] if s["name"] == scenario)

    return ExperimentConfig(
        imputer_name=imputer,
        imputer_cfg=imputer_cfg,
        scenario=parse_scenario(scenario_dict),
        rate=rate,
        seed=seed,
    )


def resolve_type(path: str) -> Any:
    """Resolve a dotted path like 'transformers.AutoModelForCausalLM.from_pretrained'."""
    parts = path.split(".")

    module = None
    attrs: list[str] = []
    for i in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:i]))
            attrs = parts[i:]
            break
        except ImportError:
            continue

    if module is None:
        raise ImportError(f"Could not import any module from path: {path}")

    result = module
    for attr in attrs:
        result = getattr(result, attr)
    return result


def parse_value(value: Any) -> Any:
    """Recursively parse YAML values, instantiating objects marked with ____type____."""
    if isinstance(value, dict):
        if "____type____" in value:
            resolved = resolve_type(value["____type____"])
            kwargs = {k: parse_value(v) for k, v in value.items() if k != "____type____"}

            # Non-callable without kwargs (e.g., torch.float16) — return as-is
            if not kwargs and not callable(resolved):
                return resolved

            return resolved(**kwargs)

        return {k: parse_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [parse_value(v) for v in value]

    return value


def instantiate_imputer(imputer_cfg: dict[str, Any]) -> Any:
    """Instantiate an imputer from YAML config using ____type____."""
    # ____pool____ is routing metadata, not a constructor arg
    cfg = {k: v for k, v in imputer_cfg.items() if k != "____pool____"}
    return parse_value(cfg)
