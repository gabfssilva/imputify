"""Cluster pool management for parallel experiment execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sky
import sky.exceptions

from experiments.object_pool import ObjectPool, PoolCallbacks, PoolConfig

if TYPE_CHECKING:
    from experiments.progress import ExperimentProgress


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    size: int
    prefix: str
    idle_timeout: int
    resources: dict[str, Any]


@dataclass(frozen=True)
class Experiment:
    imputer: str
    scenario: str
    rate: float
    seed: int
    pool: str


@dataclass
class ExperimentResult:
    experiment: Experiment
    returncode: int
    stdout: str
    stderr: str
    cached: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0


def _cache_path(output_dir: Path, exp: Experiment) -> Path:
    """Build the expected result file path for an experiment."""
    return output_dir / f"{exp.imputer}_{exp.scenario}_{exp.rate}_{exp.seed}.json"


def _check_cache(output_dir: Path | None, exp: Experiment) -> ExperimentResult | None:
    """Return cached result if exists, None otherwise."""
    if output_dir is None:
        return None

    path = _cache_path(output_dir, exp)
    if not path.exists():
        return None

    return ExperimentResult(
        experiment=exp,
        returncode=0,
        stdout="",
        stderr="",
        cached=True,
    )


def _build_resources(config: dict[str, Any]) -> sky.Resources:
    """Build sky.Resources from a pools.yaml resource config dict."""
    kwargs = dict(config)
    cloud_str = kwargs.pop("cloud", None)
    if cloud_str:
        kwargs["cloud"] = sky.CLOUD_REGISTRY[cloud_str.lower()]
    return sky.Resources(**kwargs)


class ClusterPool:
    """Pool of SkyPilot clusters using the SkyPilot Python SDK."""

    def __init__(self, config: ClusterConfig, setup: str, workdir: str):
        self.config = config
        self.setup = setup
        self.workdir = workdir
        self.cluster_names = [f"{config.prefix}-{i}" for i in range(config.size)]
        self._resources: sky.Resources = _build_resources(config.resources)
        self._progress: ExperimentProgress | None = None
        self._pool: ObjectPool[str] | None = None

    async def _create_cluster(self, name: str) -> str:
        """Launch a SkyPilot cluster via the SDK."""
        def _launch() -> str:
            task = sky.Task(
                run="echo 'Cluster ready'",
                setup=self.setup,
                workdir=self.workdir,
            )
            task.set_resources(self._resources)
            request_id = sky.launch(
                task,
                cluster_name=name,
                idle_minutes_to_autostop=self.config.idle_timeout,
                down=True,
                retry_until_up=True,
            )
            sky.get(request_id)
            return name

        return await asyncio.to_thread(_launch)

    async def _destroy_cluster(self, name: str, _value: str | None) -> None:
        """Shutdown a SkyPilot cluster via the SDK."""
        def _down() -> None:
            try:
                request_id = sky.down(name)
                sky.get(request_id)
            except sky.exceptions.ClusterDoesNotExist:
                pass

        await asyncio.to_thread(_down)

    async def _check_cluster(self, name: str, _value: str) -> bool:
        """Check if a cluster is still alive via the SDK."""
        def _check() -> bool:
            try:
                request_id = sky.status(cluster_names=[name])
                statuses = sky.get(request_id)
                return any(s.status == sky.ClusterStatus.UP for s in statuses)
            except sky.exceptions.ClusterDoesNotExist:
                return False

        return await asyncio.to_thread(_check)

    async def start(self, progress: ExperimentProgress | None = None) -> None:
        """Start the cluster pool."""
        self._progress = progress

        callbacks = PoolCallbacks[str](
            create=self._create_cluster,
            destroy=self._destroy_cluster,
            check=self._check_cluster,
            on_creating=lambda k: progress.start_cluster_launch(self.config.name, k) if progress else None,
            on_ready=lambda k: progress.cluster_ready(self.config.name, k) if progress else None,
            on_unhealthy=lambda k, e: progress.cluster_failed(self.config.name, k, e) if progress else None,
            on_stopping=lambda k: progress.cluster_stopping(self.config.name, k) if progress else None,
            on_destroyed=lambda k: progress.cluster_stopped(self.config.name, k) if progress else None,
        )

        pool_config = PoolConfig(
            min_size=self.config.size,
            max_size=self.config.size,
            max_create_attempts=3,
            health_check_interval=60.0,
            create_retry_delay=15.0,
        )

        pool = ObjectPool(
            keys=self.cluster_names,
            callbacks=callbacks,
            config=pool_config,
        )
        self._pool = pool

        await pool.start()

    async def acquire(self) -> str:
        """Acquire a cluster from the pool."""
        if self._pool is None:
            raise RuntimeError("Pool not started")
        key, _ = await self._pool.acquire()
        return key

    async def release(self, cluster: str, healthy: bool = True) -> None:
        """Release a cluster back to the pool."""
        if self._pool is None:
            return
        await self._pool.release(cluster, healthy)

    async def exec(self, cluster: str, exp: Experiment) -> ExperimentResult:
        """Execute an experiment on a cluster via the SDK."""
        cmd = (
            f"~/.local/bin/uv run python -m experiments.runner "
            f"{exp.imputer} {exp.scenario} {exp.rate} {exp.seed}"
        )

        def _exec() -> ExperimentResult:
            task = sky.Task(run=cmd)
            if "accelerators" in self.config.resources:
                task.set_resources(
                    sky.Resources(accelerators=self.config.resources["accelerators"])
                )
            try:
                request_id = sky.exec(task, cluster_name=cluster)
                sky.get(request_id)
                return ExperimentResult(
                    experiment=exp, returncode=0, stdout="", stderr="",
                )
            except (
                sky.exceptions.ClusterDoesNotExist,
                sky.exceptions.ClusterNotUpError,
            ) as e:
                return ExperimentResult(
                    experiment=exp, returncode=1, stdout="",
                    stderr=f"ClusterDoesNotExist: {e}",
                )
            except sky.exceptions.CommandError as e:
                return ExperimentResult(
                    experiment=exp, returncode=e.returncode, stdout="",
                    stderr=e.detailed_reason or e.error_msg,
                )
            except Exception as e:
                return ExperimentResult(
                    experiment=exp, returncode=1, stdout="", stderr=str(e),
                )

        return await asyncio.to_thread(_exec)

    async def shutdown(self) -> None:
        """Shutdown the pool."""
        if self._pool:
            await self._pool.close()


class PoolManager:
    """Manages multiple cluster pools."""

    def __init__(self, pools_config: dict[str, Any]):
        self.pools: dict[str, ClusterPool] = {}
        workdir = pools_config.get("workdir", ".")

        for name, cfg in pools_config.get("pools", {}).items():
            pool_cfg = ClusterConfig(
                name=name,
                size=cfg["size"],
                prefix=cfg["prefix"],
                idle_timeout=cfg.get("idle_timeout", 10),
                resources=cfg["resources"],
            )
            setup = cfg.get("setup", "")
            self.pools[name] = ClusterPool(pool_cfg, setup, workdir)

    async def start(
        self,
        pool_names: list[str] | None = None,
        progress: ExperimentProgress | None = None,
    ) -> None:
        """Start launching pools."""
        names = pool_names or list(self.pools.keys())
        pools = [self.pools[n] for n in names if n in self.pools]

        await asyncio.gather(*[p.start(progress) for p in pools])

    async def run(
        self,
        experiments: list[Experiment],
        progress: ExperimentProgress | None = None,
        output_dir: Path | None = None,
        max_retries: int = 2,
    ) -> list[ExperimentResult]:
        """Execute experiments on appropriate pools with retry on failure."""

        async def rsync_from_cluster(cluster: str) -> None:
            """Rsync results from a single cluster."""
            if output_dir is None:
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                "rsync", "-avz",
                f"{cluster}:~/sky_workdir/outputs/",
                str(output_dir) + "/",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

        async def run_one(exp: Experiment) -> ExperimentResult:
            cached = _check_cache(output_dir, exp)
            if cached:
                if progress:
                    progress.experiment_cached(exp.pool, exp)
                return cached

            pool = self.pools[exp.pool]

            for attempt in range(max_retries + 1):
                cluster = await pool.acquire()
                try:
                    if progress:
                        progress.start_experiment(exp.pool, cluster, exp)

                    result = await pool.exec(cluster, exp)

                    # Check if cluster died (preemption)
                    is_healthy = "ClusterDoesNotExist" not in result.stderr
                    await pool.release(cluster, healthy=is_healthy)

                    if not is_healthy and attempt < max_retries:
                        continue

                    if progress:
                        progress.complete_experiment(exp.pool, cluster, exp, result.success)

                    if result.success:
                        await rsync_from_cluster(cluster)

                    return result

                except Exception:
                    await pool.release(cluster, healthy=False)
                    if attempt == max_retries:
                        raise

            raise RuntimeError("Unexpected: all retries exhausted")

        return list(await asyncio.gather(*[run_one(exp) for exp in experiments]))

    async def shutdown(self) -> None:
        """Shutdown all pools."""
        await asyncio.gather(*[p.shutdown() for p in self.pools.values()])
