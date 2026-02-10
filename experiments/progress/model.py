"""Pure state model for experiment progress — no Rich imports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.pool import Experiment

__all__ = [
    "ClusterStatus",
    "RateNode",
    "MechanismNode",
    "DatasetNode",
    "ImputerNode",
    "ClusterInfo",
    "PoolInfo",
    "ExperimentModel",
]


class ClusterStatus(Enum):
    PENDING = "pending"
    LAUNCHING = "launching"
    READY = "ready"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class RateNode:
    rate: float
    total_seeds: int = 0
    completed: int = 0
    running: int = 0
    failed: int = 0

    @property
    def is_active(self) -> bool:
        return self.running > 0

    @property
    def is_complete(self) -> bool:
        return self.total_seeds > 0 and (self.completed + self.failed) == self.total_seeds

    @property
    def pending(self) -> int:
        return self.total_seeds - self.completed - self.running - self.failed


@dataclass
class MechanismNode:
    name: str
    rates: dict[float, RateNode] = field(default_factory=dict)

    @property
    def total_seeds(self) -> int:
        return sum(r.total_seeds for r in self.rates.values())

    @property
    def completed(self) -> int:
        return sum(r.completed for r in self.rates.values())

    @property
    def running(self) -> int:
        return sum(r.running for r in self.rates.values())

    @property
    def failed(self) -> int:
        return sum(r.failed for r in self.rates.values())

    @property
    def is_active(self) -> bool:
        return self.running > 0

    @property
    def is_complete(self) -> bool:
        return self.total_seeds > 0 and (self.completed + self.failed) == self.total_seeds


@dataclass
class DatasetNode:
    name: str
    mechanisms: dict[str, MechanismNode] = field(default_factory=dict)

    @property
    def total_seeds(self) -> int:
        return sum(m.total_seeds for m in self.mechanisms.values())

    @property
    def completed(self) -> int:
        return sum(m.completed for m in self.mechanisms.values())

    @property
    def running(self) -> int:
        return sum(m.running for m in self.mechanisms.values())

    @property
    def failed(self) -> int:
        return sum(m.failed for m in self.mechanisms.values())

    @property
    def is_active(self) -> bool:
        return self.running > 0

    @property
    def is_complete(self) -> bool:
        return self.total_seeds > 0 and (self.completed + self.failed) == self.total_seeds


@dataclass
class ImputerNode:
    name: str
    datasets: dict[str, DatasetNode] = field(default_factory=dict)

    @property
    def total_seeds(self) -> int:
        return sum(d.total_seeds for d in self.datasets.values())

    @property
    def completed(self) -> int:
        return sum(d.completed for d in self.datasets.values())

    @property
    def running(self) -> int:
        return sum(d.running for d in self.datasets.values())

    @property
    def failed(self) -> int:
        return sum(d.failed for d in self.datasets.values())

    @property
    def is_active(self) -> bool:
        return self.running > 0

    @property
    def is_complete(self) -> bool:
        return self.total_seeds > 0 and (self.completed + self.failed) == self.total_seeds


@dataclass
class ClusterInfo:
    name: str
    status: ClusterStatus = ClusterStatus.PENDING
    error: str | None = None
    launch_start_time: float | None = None

    @property
    def launch_elapsed(self) -> str:
        if self.launch_start_time is None:
            return ""
        secs = int(time.time() - self.launch_start_time)
        mins, secs = divmod(secs, 60)
        return f"{mins}m {secs:02d}s"


@dataclass
class PoolInfo:
    name: str
    clusters: dict[str, ClusterInfo] = field(default_factory=dict)
    hourly_cost: float = 0.0


def _parse_scenario(scenario: str) -> tuple[str, str]:
    """Parse 'dataset-MECHANISM' into (dataset, mechanism).

    >>> _parse_scenario('iris-MCAR')
    ('iris', 'MCAR')
    >>> _parse_scenario('breast_cancer-MAR')
    ('breast_cancer', 'MAR')
    """
    dataset, mechanism = scenario.rsplit("-", 1)
    return dataset, mechanism


class ExperimentModel:
    """Central state for all experiment progress."""

    def __init__(self) -> None:
        self.imputers: dict[str, ImputerNode] = {}
        self.pools: dict[str, PoolInfo] = {}
        self.total: int = 0
        self.running: int = 0
        self.completed: int = 0
        self.failed: int = 0
        self.cached: int = 0
        self.start_time: float | None = None
        self._first_complete_time: float | None = None
        self._last_complete_time: float | None = None
        self._completed_since_first: int = 0

    @property
    def pending(self) -> int:
        return self.total - self.running - self.completed - self.failed - self.cached

    @property
    def done(self) -> int:
        return self.completed + self.cached

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> float | None:
        """ETA based on throughput since first completion."""
        if self._first_complete_time is None or self._completed_since_first < 2:
            return None
        elapsed = time.time() - self._first_complete_time
        rate = self._completed_since_first / elapsed
        remaining = self.total - self.done
        if rate <= 0:
            return None
        return remaining / rate

    @property
    def total_hourly_cost(self) -> float:
        """Sum of hourly costs for all ready clusters."""
        total = 0.0
        for pool_info in self.pools.values():
            for cluster in pool_info.clusters.values():
                if cluster.status == ClusterStatus.READY:
                    total += pool_info.hourly_cost
        return total

    @property
    def estimated_cost(self) -> float:
        """Estimated total cost based on elapsed time and active clusters."""
        cost = 0.0
        for pool_info in self.pools.values():
            ready_count = sum(
                1 for c in pool_info.clusters.values()
                if c.status in (ClusterStatus.READY, ClusterStatus.STOPPING, ClusterStatus.STOPPED)
            )
            cost += ready_count * pool_info.hourly_cost * (self.elapsed / 3600)
        return cost

    def register_experiments(self, experiments: list[Experiment]) -> None:
        """Build the full imputer -> dataset -> mechanism -> rate tree."""
        self.total = len(experiments)
        self.start_time = time.time()

        for exp in experiments:
            dataset, mechanism = _parse_scenario(exp.scenario)

            # Ensure imputer node
            if exp.imputer not in self.imputers:
                self.imputers[exp.imputer] = ImputerNode(name=exp.imputer)
            imp_node = self.imputers[exp.imputer]

            # Ensure dataset node
            if dataset not in imp_node.datasets:
                imp_node.datasets[dataset] = DatasetNode(name=dataset)
            ds_node = imp_node.datasets[dataset]

            # Ensure mechanism node
            if mechanism not in ds_node.mechanisms:
                ds_node.mechanisms[mechanism] = MechanismNode(name=mechanism)
            mech_node = ds_node.mechanisms[mechanism]

            # Ensure rate node
            if exp.rate not in mech_node.rates:
                mech_node.rates[exp.rate] = RateNode(rate=exp.rate)
            mech_node.rates[exp.rate].total_seeds += 1

    def register_pool(
        self, name: str, cluster_names: list[str], hourly_cost: float = 0.0
    ) -> None:
        """Register a pool with its cluster names and per-cluster hourly cost."""
        self.pools[name] = PoolInfo(
            name=name,
            clusters={c: ClusterInfo(name=c) for c in cluster_names},
            hourly_cost=hourly_cost,
        )

    # -- Cluster callbacks --

    def start_cluster_launch(self, pool: str, cluster: str) -> None:
        info = self.pools[pool].clusters[cluster]
        info.status = ClusterStatus.LAUNCHING
        info.launch_start_time = time.time()

    def cluster_ready(self, pool: str, cluster: str) -> None:
        self.pools[pool].clusters[cluster].status = ClusterStatus.READY

    def cluster_failed(self, pool: str, cluster: str, error: str = "") -> None:
        info = self.pools[pool].clusters[cluster]
        info.status = ClusterStatus.FAILED
        info.error = error

    def cluster_stopping(self, pool: str, cluster: str) -> None:
        info = self.pools[pool].clusters[cluster]
        info.status = ClusterStatus.STOPPING
        info.launch_start_time = time.time()

    def cluster_stopped(self, pool: str, cluster: str) -> None:
        self.pools[pool].clusters[cluster].status = ClusterStatus.STOPPED

    # -- Experiment callbacks --

    def start_experiment(self, exp: Experiment) -> None:
        """Mark an experiment as running."""
        rate_node = self._find_rate_node(exp)
        rate_node.running += 1
        self.running += 1

    def complete_experiment(self, exp: Experiment, success: bool) -> None:
        """Mark an experiment as completed or failed."""
        rate_node = self._find_rate_node(exp)
        rate_node.running -= 1
        self.running -= 1

        now = time.time()
        if self._first_complete_time is None:
            self._first_complete_time = now
        self._last_complete_time = now
        self._completed_since_first += 1

        if success:
            rate_node.completed += 1
            self.completed += 1
        else:
            rate_node.failed += 1
            self.failed += 1

    def experiment_cached(self, exp: Experiment) -> None:
        """Mark an experiment as served from cache."""
        rate_node = self._find_rate_node(exp)
        rate_node.completed += 1
        self.cached += 1

    def _find_rate_node(self, exp: Experiment) -> RateNode:
        """O(1) lookup for the rate node of an experiment."""
        dataset, mechanism = _parse_scenario(exp.scenario)
        return (
            self.imputers[exp.imputer]
            .datasets[dataset]
            .mechanisms[mechanism]
            .rates[exp.rate]
        )
