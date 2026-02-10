"""Experiment runner: fits imputers, evaluates metrics, saves results."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import cyclopts
import numpy as np

from experiments.config import ExperimentConfig, build_config, instantiate_imputer, load_experiments
from experiments.datasets import load_dataset
from imputify import evaluate
from imputify.missing import introduce_missing
from imputify.shared import cleanup_gpu, seed_everything

app = cyclopts.App(help="Run imputation experiments")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a single experiment and return results."""
    cleanup_gpu()
    seed_everything(config.seed)

    X, y = load_dataset(config.scenario.dataset)

    X_miss, mask = introduce_missing(
        X,
        proportion=config.rate,
        patterns=list(config.scenario.patterns),
        seed=config.seed,
    )

    imputer = instantiate_imputer(config.imputer_cfg)
    imputer.fit(X_miss)
    X_imputed = imputer.transform(X_miss)

    inference_results = []
    for inf_seed in (42, 21, 84):
        seed_everything(inf_seed)
        metrics = evaluate(
            X=X,
            X_imputed=X_imputed,
            missing_mask=mask,
            y=y.to_numpy(),
            X_missing=X_miss,
        )
        inference_results.append({
            "inference_seed": inf_seed,
            "overall_score": metrics.overall_score,
            "reconstruction_score": metrics.overall_reconstruction_score,
            "distribution_score": metrics.overall_distribution_score,
            "predictive_score": metrics.overall_predictive_score,
            "metrics": metrics.to_dict(),
        })

    return {
        "imputer": config.imputer_name,
        "scenario": config.scenario.name,
        "dataset": config.scenario.dataset,
        "rate": config.rate,
        "seed": config.seed,
        "inference_results": inference_results,
        "avg_score": float(np.mean([r["overall_score"] for r in inference_results])),
    }


def _resolve_seeds(
    experiments: dict[str, Any], imputer: str, seed: int | None, *, fast: bool = False
) -> list[int]:
    if seed is not None:
        return [seed]
    is_gpu = experiments["imputers"][imputer].get("____pool____") == "gpu"
    seeds = experiments["llm_seeds"] if is_gpu else experiments["seeds"]
    if fast:
        return seeds[:1] if is_gpu else seeds[:3]
    return seeds


def _run_single(
    experiments: dict[str, Any],
    imputer: str,
    scenario: str,
    rate: float,
    seed: int,
    output_dir: Path,
    prefix: str = "",
) -> None:
    exp_config = build_config(experiments, imputer, scenario, rate, seed)
    print(f"{prefix}{scenario} | rate={rate} | seed={seed}", end=" ... ", flush=True)
    result = run_experiment(exp_config)
    print(f"score={result['avg_score']:.4f}")
    output_file = output_dir / f"{imputer}_{scenario}_{rate}_{seed}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


@app.default
def main(
    imputer: str,
    scenario: str | None = None,
    rate: float | None = None,
    seed: int | None = None,
    config: Path = Path(__file__).parent / "experiments.yaml",
    output_dir: Path = Path.home() / "sky_workdir" / "outputs",
    fast: bool = False,
) -> None:
    """Run imputation experiments. Omit scenario/rate/seed to run all."""
    experiments = load_experiments(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [scenario] if scenario else [s["name"] for s in experiments["scenarios"]]
    rates = [rate] if rate else experiments["missing_rates"]
    seeds = _resolve_seeds(experiments, imputer, seed, fast=fast)

    combinations = list(product(scenarios, rates, seeds))
    total = len(combinations)

    print(f"Running {imputer}: {len(scenarios)} scenarios x {len(rates)} rates x {len(seeds)} seeds = {total} experiments\n")

    for i, (s, r, sd) in enumerate(combinations, 1):
        _run_single(experiments, imputer, s, r, sd, output_dir, prefix=f"[{i}/{total}] ")


if __name__ == "__main__":
    app()
