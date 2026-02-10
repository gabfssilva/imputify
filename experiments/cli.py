"""CLI orchestrator for parallel experiment execution."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import cyclopts
import yaml

from experiments.pool import Experiment, PoolManager
from experiments.progress import ExperimentProgress

app = cyclopts.App(help="Orchestrate imputation experiments across cluster pools")

CONFIG_DIR = Path(__file__).parent


def load_pools_config() -> dict[str, Any]:
    """Load pools configuration."""
    return yaml.safe_load((CONFIG_DIR / "pools.yaml").read_text())


def load_experiments_config() -> dict[str, Any]:
    """Load experiments configuration."""
    return yaml.safe_load((CONFIG_DIR / "experiments.yaml").read_text())


def generate_experiments(
    experiments_cfg: dict[str, Any],
    imputers: list[str] | None = None,
    scenarios: list[str] | None = None,
    pools: list[str] | None = None,
    *,
    fast: bool = False,
) -> list[Experiment]:
    """Generate experiment matrix with optional filters."""
    result = []

    imputer_cfgs = experiments_cfg["imputers"]
    imputer_names = imputers or list(imputer_cfgs.keys())
    scenario_names = scenarios or [s["name"] for s in experiments_cfg["scenarios"]]

    cpu_seeds = experiments_cfg["seeds"][:3] if fast else experiments_cfg["seeds"]
    llm_seeds = experiments_cfg["llm_seeds"][:1] if fast else experiments_cfg["llm_seeds"]

    for imp_name in imputer_names:
        if imp_name not in imputer_cfgs:
            print(f"Warning: imputer '{imp_name}' not found, skipping")
            continue

        imp_cfg = imputer_cfgs[imp_name]
        pool = imp_cfg.get("____pool____", "cpu")

        if pools and pool not in pools:
            continue

        seeds = llm_seeds if pool == "gpu" else cpu_seeds

        for scenario in scenario_names:
            for rate in experiments_cfg["missing_rates"]:
                for seed in seeds:
                    result.append(Experiment(
                        imputer=imp_name,
                        scenario=scenario,
                        rate=rate,
                        seed=seed,
                        pool=pool,
                    ))

    return result


@app.command()
def run(
    imputer: list[str] | None = None,
    scenario: list[str] | None = None,
    pool: list[str] | None = None,
    output_dir: Path = Path("./results"),
    dry_run: bool = False,
    no_shutdown: bool = False,
    fast: bool = False,
    cpu_nodes: int | None = None,
    gpu_nodes: int | None = None,
) -> None:
    """Execute experiments in parallel across cluster pools.

    Args:
        imputer: Filter by imputer names (can specify multiple)
        scenario: Filter by scenario names (can specify multiple)
        pool: Filter by pool names (can specify multiple)
        output_dir: Directory to save results
        dry_run: Print experiments without running
        no_shutdown: Keep clusters running after completion
        fast: Use fewer seeds (CPU: 3, GPU: 1)
        cpu_nodes: Override number of CPU cluster nodes
        gpu_nodes: Override number of GPU cluster nodes
    """
    pools_cfg = load_pools_config()
    if cpu_nodes is not None and "cpu" in pools_cfg.get("pools", {}):
        pools_cfg["pools"]["cpu"]["size"] = cpu_nodes
    if gpu_nodes is not None and "gpu" in pools_cfg.get("pools", {}):
        pools_cfg["pools"]["gpu"]["size"] = gpu_nodes
    experiments_cfg = load_experiments_config()

    experiments = generate_experiments(experiments_cfg, imputer, scenario, pool, fast=fast)

    if not experiments:
        print("No experiments to run")
        return

    by_pool: dict[str, int] = {}
    for exp in experiments:
        by_pool[exp.pool] = by_pool.get(exp.pool, 0) + 1

    if dry_run:
        print("\nExperiments:")
        for exp in experiments[:20]:
            print(f"  {exp.imputer} | {exp.scenario} | {exp.rate} | {exp.seed} -> {exp.pool}")
        if len(experiments) > 20:
            print(f"  ... and {len(experiments) - 20} more")
        return

    print()

    # Redirect all logging to file so Rich Live display isn't disrupted
    log_file = output_dir / "experiments.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_file),
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    async def main() -> None:
        manager = PoolManager(pools_cfg)

        needed_pools = list(set(exp.pool for exp in experiments))

        progress = ExperimentProgress()
        progress.register_experiments(experiments)
        for pool_name in needed_pools:
            pool = manager.pools[pool_name]
            progress.register_pool(pool_name, pool.cluster_names, pool.config.resources)

        results: list = []
        try:
            with progress:
                await manager.start(needed_pools, progress)
                results = await manager.run(experiments, progress, output_dir)

                if not no_shutdown:
                    await manager.shutdown()

        except KeyboardInterrupt:
            print("\n\nInterrupted! Shutting down clusters...")
            if not no_shutdown:
                with progress:
                    await manager.shutdown()
            return

        except Exception:
            if not no_shutdown:
                with progress:
                    await manager.shutdown()
            raise

        failed = [r for r in results if not r.success]
        if failed:
            print("\nFailed experiments:")
            for r in failed:
                exp = r.experiment
                print(f"  {exp.imputer} | {exp.scenario} | {exp.rate} | {exp.seed}")
                print(f"    stderr: {r.stderr[:200]}")

        if no_shutdown:
            print("\nClusters kept running (--no-shutdown)")

    asyncio.run(main())


@app.command()
def status() -> None:
    """Show status of all clusters."""
    import sky

    request_id = sky.status()
    statuses = sky.get(request_id)
    if not statuses:
        print("No clusters found.")
        return
    for s in statuses:
        print(f"  {s.name}: {s.status.value}")


@app.command()
def down() -> None:
    """Terminate all RunPod pods."""
    import runpod

    pods = runpod.get_pods()
    if not pods:
        print("No pods running.")
        return

    print(f"Terminating {len(pods)} pods...")
    for p in pods:
        try:
            runpod.terminate_pod(p["id"])
            print(f"  {p['name']} terminated")
        except runpod.error.QueryError:
            print(f"  {p['name']} already gone")

    print("Done.")


@app.command()
def list_experiments(
    imputer: list[str] | None = None,
    scenario: list[str] | None = None,
    pool: list[str] | None = None,
    fast: bool = False,
) -> None:
    """List all experiments that would run.

    Args:
        imputer: Filter by imputer names
        scenario: Filter by scenario names
        pool: Filter by pool names
        fast: Use fewer seeds (CPU: 3, GPU: 1)
    """
    experiments_cfg = load_experiments_config()
    experiments = generate_experiments(experiments_cfg, imputer, scenario, pool, fast=fast)

    by_pool: dict[str, list[Experiment]] = {}
    for exp in experiments:
        by_pool.setdefault(exp.pool, []).append(exp)

    for pool_name, exps in by_pool.items():
        print(f"\n{pool_name} ({len(exps)} experiments):")
        imputers = set(e.imputer for e in exps)
        scenarios = set(e.scenario for e in exps)
        rates = set(e.rate for e in exps)
        seeds = set(e.seed for e in exps)

        print(f"  imputers:  {', '.join(sorted(imputers))}")
        print(f"  scenarios: {', '.join(sorted(scenarios))}")
        print(f"  rates:     {', '.join(str(r) for r in sorted(rates))}")
        print(f"  seeds:     {', '.join(str(s) for s in sorted(seeds))}")


if __name__ == "__main__":
    app()
