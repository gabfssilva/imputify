# Experiments

Evaluates imputers across the cross-product: imputers x datasets x missing mechanisms (MCAR/MAR/MNAR) x rates x seeds. Each experiment loads a dataset, introduces missing values, fits the imputer, and computes reconstruction, distribution, and predictive metrics.

Full matrix defined in `experiments.yaml`. SkyPilot cluster pools in `pools.yaml`.

## Running locally

```bash
# Single experiment
uv run python -m experiments.runner dae_h128 --scenario iris-MCAR --rate 0.2 --seed 42

# All scenarios/rates/seeds for one imputer
uv run python -m experiments.runner statistical_mean

# Filter by scenario or rate
uv run python -m experiments.runner vae_h128 --scenario diabetes-MAR
uv run python -m experiments.runner vae_h128 --rate 0.5

# Fewer seeds for quick iteration (CPU: 3, GPU: 1)
uv run python -m experiments.runner statistical_mean --fast

# Custom output directory
uv run python -m experiments.runner knn_k5 --output-dir ./my-results
```

## Running on SkyPilot clusters

```bash
# Everything
uv run python -m experiments.cli run

# Filters (combinable, accepts multiple values)
uv run python -m experiments.cli run --imputer dae_h128 --imputer vae_h128
uv run python -m experiments.cli run --scenario iris-MCAR
uv run python -m experiments.cli run --pool gpu

# Fewer seeds for quick iteration (CPU: 3, GPU: 1)
uv run python -m experiments.cli run --fast

# Preview without executing
uv run python -m experiments.cli run --dry-run

# List experiment matrix
uv run python -m experiments.cli list-experiments
uv run python -m experiments.cli list-experiments --imputer gain_h128

# Cluster management
uv run python -m experiments.cli status
uv run python -m experiments.cli down          # shutdown all pools
uv run python -m experiments.cli down --pool cpu
```

Results are cached: experiments with an existing JSON file in `--output-dir` are skipped on re-runs.
