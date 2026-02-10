"""Controller: thin wrapper managing model + view + Rich Live lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live

from experiments.progress.model import ExperimentModel
from experiments.progress.view import ProgressView

if TYPE_CHECKING:
    from experiments.pool import Experiment

__all__ = ["ExperimentProgress"]


class _LivePanel:
    """Rich renderable that produces the current progress panel on each render cycle."""

    def __init__(self, view: ProgressView, model: ExperimentModel) -> None:
        self._view = view
        self._model = model

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self._view.render(self._model)


def _resolve_hourly_cost(resources: dict[str, object]) -> float:
    """Resolve hourly cost for a pool via SkyPilot catalog."""
    try:
        import sky

        cloud = sky.RunPod() if resources.get("cloud") == "runpod" else None
        r = sky.Resources(
            cloud=cloud,
            instance_type=resources.get("instance_type"),  # type: ignore[arg-type]
            accelerators=resources.get("accelerators"),  # type: ignore[arg-type]
            use_spot=resources.get("use_spot", False),  # type: ignore[arg-type]
        )
        return r.get_cost(3600)
    except Exception:
        return 0.0


class ExperimentProgress:
    """Live progress display for experiments.

    Public API: mutate model, then refresh view.
    """

    def __init__(self) -> None:
        self._model = ExperimentModel()
        self._view = ProgressView()
        self._console = Console()
        self._live: Live | None = None

    # -- Registration (called from cli.py) --

    def register_experiments(self, experiments: list[Experiment]) -> None:
        """Build the full imputer tree from experiments list."""
        self._model.register_experiments(experiments)
        self._refresh()

    def register_pool(
        self,
        name: str,
        cluster_names: list[str],
        resources: dict[str, object] | None = None,
    ) -> None:
        """Register a cluster pool with optional resource config for cost estimation."""
        hourly_cost = _resolve_hourly_cost(resources) if resources else 0.0
        self._model.register_pool(name, cluster_names, hourly_cost=hourly_cost)
        self._refresh()

    # -- Cluster callbacks (called from pool.py ObjectPool) --

    def start_cluster_launch(self, pool: str, cluster: str) -> None:
        self._model.start_cluster_launch(pool, cluster)
        self._refresh()

    def cluster_ready(self, pool: str, cluster: str) -> None:
        self._model.cluster_ready(pool, cluster)
        self._refresh()

    def cluster_failed(self, pool: str, cluster: str, error: str = "") -> None:
        self._model.cluster_failed(pool, cluster, error)
        self._refresh()

    def cluster_stopping(self, pool: str, cluster: str) -> None:
        self._model.cluster_stopping(pool, cluster)
        self._refresh()

    def cluster_stopped(self, pool: str, cluster: str) -> None:
        self._model.cluster_stopped(pool, cluster)
        self._refresh()

    # -- Experiment callbacks (called from pool.py PoolManager) --

    def start_experiment(self, pool: str, cluster: str, exp: Experiment) -> None:
        """Mark experiment as started on cluster."""
        _ = pool, cluster  # kept for API compat with callers
        self._model.start_experiment(exp)
        self._refresh()

    def complete_experiment(
        self, pool: str, cluster: str, exp: Experiment, success: bool
    ) -> None:
        """Mark experiment as completed."""
        _ = pool, cluster
        self._model.complete_experiment(exp, success)
        self._refresh()

    def experiment_cached(self, pool: str, exp: Experiment) -> None:
        """Called when experiment was served from cache."""
        _ = pool
        self._model.experiment_cached(exp)
        self._refresh()

    # -- Live lifecycle --

    def _refresh(self) -> None:
        if self._live:
            self._live.refresh()

    def __enter__(self) -> ExperimentProgress:
        self._live = Live(
            _LivePanel(self._view, self._model),
            console=self._console,
            refresh_per_second=4,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(None, None, None)
