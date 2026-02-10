"""Pure rendering for experiment progress — Rich display with compaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.bar import Bar
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from rich.console import RenderableType

from experiments.progress.model import (
    ClusterStatus,
    DatasetNode,
    ExperimentModel,
    ImputerNode,
    MechanismNode,
    RateNode,
)

__all__ = ["ProgressView"]

SPINNER_FRAMES = ("◐", "◓", "◑", "◒")
MAX_PENDING_DATASETS = 3


class ProgressView:
    """Renders experiment progress from model state.

    Only mutable state is ``_spinner_idx``, incremented per render for animation.
    """

    def __init__(self) -> None:
        self._spinner_idx: int = 0

    @property
    def _spinner(self) -> str:
        return SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]

    def render(self, model: ExperimentModel) -> Panel:
        """Build the full display panel from model state."""
        self._spinner_idx += 1

        parts = Text()

        # Cluster bar
        cluster_bar = self._render_cluster_bar(model)
        if cluster_bar:
            parts.append_text(cluster_bar)
            parts.append("\n\n")

        # Imputers — sorted: active first, pending, complete
        sorted_imputers = sorted(
            model.imputers.values(),
            key=lambda n: (0 if n.is_active else (1 if not n.is_complete else 2), n.name),
        )

        tree = Tree("")
        for imp_node in sorted_imputers:
            self._render_imputer(tree, imp_node)

        # Footer
        footer = self._render_footer(model)

        renderables: list[RenderableType] = []
        if cluster_bar:
            renderables.append(cluster_bar)
            renderables.append(Text(""))
        renderables.append(tree)
        renderables.extend(footer)

        return Panel(
            Group(*renderables),
            title="[bold blue]Experiments[/bold blue]",
            border_style="blue",
        )

    def _render_cluster_bar(self, model: ExperimentModel) -> Text:
        """Render: Clusters: cpu 5/10 ◓  |  gpu 18/20 ●"""
        if not model.pools:
            return Text()

        parts = Text("Clusters: ")
        pool_items: list[Text] = []

        for pool_info in model.pools.values():
            ready = sum(
                1 for c in pool_info.clusters.values()
                if c.status == ClusterStatus.READY
            )
            total = len(pool_info.clusters)
            launching = any(
                c.status == ClusterStatus.LAUNCHING
                for c in pool_info.clusters.values()
            )

            item = Text(f"{pool_info.name} {ready}/{total} ")
            if launching:
                item.append(self._spinner, style="yellow")
            elif ready == total and total > 0:
                item.append("●", style="green")
            elif ready > 0:
                item.append("◐", style="dim")
            else:
                item.append("○", style="dim")

            pool_items.append(item)

        for i, item in enumerate(pool_items):
            if i > 0:
                parts.append("  |  ")
            parts.append_text(item)

        return parts

    def _render_imputer(self, tree: Tree, node: ImputerNode) -> None:
        """Render an imputer: collapsed if pending/complete, expanded if active."""
        counter = f"{node.completed}/{node.total_seeds}"

        if node.is_active:
            icon = f"[yellow]{self._spinner}[/yellow]"
            label = f"{icon} {node.name}  [dim]{counter}[/dim]"
            branch = tree.add(label)
            self._render_imputer_datasets(branch, node)
        elif node.is_complete:
            if node.failed > 0:
                icon = "[yellow]![/yellow]"
            else:
                icon = "[green]✓[/green]"
            tree.add(f"{icon} {node.name}  [dim]{counter}[/dim]")
        else:
            tree.add(f"[dim]○[/dim] {node.name}  [dim]{counter}[/dim]")

    def _render_imputer_datasets(self, branch: Tree, node: ImputerNode) -> None:
        """Expand datasets for an active imputer with compaction."""
        sorted_datasets = sorted(
            node.datasets.values(),
            key=lambda d: (0 if d.is_active else (1 if not d.is_complete else 2), d.name),
        )

        pending_count = 0
        shown = 0

        for ds in sorted_datasets:
            if ds.is_active:
                # Active dataset — expand with mechanisms
                self._render_dataset_expanded(branch, ds)
                shown += 1
            elif ds.is_complete:
                # Complete — one line
                if ds.failed > 0:
                    icon = "[yellow]![/yellow]"
                else:
                    icon = "[green]✓[/green]"
                branch.add(f"{icon} {ds.name}")
                shown += 1
            else:
                # Pending — show first few, then collapse
                if pending_count < MAX_PENDING_DATASETS:
                    branch.add(f"[dim]○[/dim] {ds.name}")
                    shown += 1
                pending_count += 1

        collapsed = pending_count - min(pending_count, MAX_PENDING_DATASETS)
        if collapsed > 0:
            branch.add(f"[dim](+ {collapsed} more pending)[/dim]")

    def _render_dataset_expanded(self, parent: Tree, ds: DatasetNode) -> None:
        """Render an active dataset with mechanism lines and rate badges."""
        icon = f"[yellow]{self._spinner}[/yellow]"
        ds_branch = parent.add(f"{icon} {ds.name}")

        sorted_mechs = sorted(
            ds.mechanisms.values(),
            key=lambda m: (0 if m.is_active else (1 if not m.is_complete else 2), m.name),
        )

        for mech in sorted_mechs:
            badges = self._render_rate_badges(mech)
            ds_branch.add(f"{mech.name}  {badges}")

    def _render_rate_badges(self, mech: MechanismNode) -> str:
        """Render inline rate badges: ✓10% ◓20% ○30%"""
        parts: list[str] = []

        for rate in sorted(mech.rates.keys()):
            rate_node = mech.rates[rate]
            pct = _format_rate(rate)
            icon = _rate_icon(rate_node, self._spinner)
            parts.append(f"{icon} {pct}")

        return "[" + " ".join(parts) + "]"

    def _render_footer(self, model: ExperimentModel) -> list[RenderableType]:
        """Render progress bar + stats + time + cost."""
        items: list[RenderableType] = []

        # Progress bar
        if model.total > 0:
            bar = Bar(
                size=model.total,
                begin=0,
                end=model.done,
                width=60,
                color="green",
                bgcolor="grey23",
            )
            pct = model.done / model.total * 100
            bar_line = Text()
            bar_line.append(f" {pct:5.1f}%  ", style="dim")
            items.append(Text(""))
            items.append(bar)
            items.append(bar_line)

        # Stats line
        stats = Text()
        stats.append(f"Done: ", style="dim")
        stats.append(f"{model.done}", style="green")
        stats.append(f"/{model.total}", style="dim")

        if model.cached > 0:
            stats.append(f"  Cached: ", style="dim")
            stats.append(str(model.cached), style="blue")

        if model.failed > 0:
            stats.append(f"  Failed: ", style="dim")
            stats.append(str(model.failed), style="red")

        stats.append(f"  Pending: ", style="dim")
        stats.append(str(model.pending))

        if model.running > 0:
            stats.append(f"  Running: ", style="dim")
            stats.append(str(model.running), style="yellow")

        items.append(stats)

        # Time + cost line
        time_line = Text()
        elapsed = model.elapsed
        if elapsed > 0:
            time_line.append("Elapsed: ", style="dim")
            time_line.append(_format_duration(elapsed))

            eta = model.eta_seconds
            if eta is not None:
                time_line.append("  ETA: ", style="dim")
                time_line.append(_format_duration(eta))

            cost = model.estimated_cost
            if cost > 0:
                time_line.append("  Cost: ", style="dim")
                time_line.append(f"${cost:.2f}", style="yellow")

                # Projected total cost
                if model.done > 0 and model.done < model.total:
                    projected = cost / model.done * model.total
                    time_line.append(f"  (est. total: ${projected:.2f})", style="dim")

            items.append(time_line)

        return items


def _rate_icon(node: RateNode, spinner: str) -> str:
    """Pick the icon for a rate badge."""
    if node.is_active:
        return f"[yellow]{spinner}[/yellow]"
    if node.is_complete:
        if node.failed > 0:
            return "[yellow]![/yellow]"
        return "[green]✓[/green]"
    if node.completed > 0:
        return "[dim]◐[/dim]"
    return "[dim]○[/dim]"


def _format_rate(rate: float) -> str:
    """Format rate as percentage: 0.2 -> '20%'"""
    pct = int(rate * 100)
    return f"{pct}%"


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m"


