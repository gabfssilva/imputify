"""Generic async object pool with automatic health checking and recovery."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ItemStatus(Enum):
    CREATING = "creating"
    READY = "ready"
    IN_USE = "in_use"
    UNHEALTHY = "unhealthy"
    DESTROYED = "destroyed"


@dataclass
class PoolItem(Generic[T]):
    """Wrapper for pool items with metadata."""

    key: str
    value: T | None = None
    status: ItemStatus = ItemStatus.CREATING
    create_attempts: int = 0


@dataclass
class PoolConfig:
    """Pool configuration."""

    min_size: int = 1
    max_size: int = 10
    max_create_attempts: int = 5
    health_check_interval: float = 30.0
    create_retry_delay: float = 10.0


@dataclass
class PoolCallbacks(Generic[T]):
    """Callbacks for pool lifecycle events."""

    create: Callable[[str], Awaitable[T]]
    destroy: Callable[[str, T | None], Awaitable[None]]
    check: Callable[[str, T], Awaitable[bool]] | None = None
    on_creating: Callable[[str], None] | None = None
    on_ready: Callable[[str], None] | None = None
    on_unhealthy: Callable[[str, str], None] | None = None
    on_stopping: Callable[[str], None] | None = None
    on_destroyed: Callable[[str], None] | None = None


class ObjectPool(Generic[T]):
    """Async object pool with automatic health checking and recovery.

    Features:
    - Automatic creation of items up to max_size
    - Health checking with automatic recovery
    - Retry on creation failures
    - Graceful shutdown
    """

    def __init__(
        self,
        keys: list[str],
        callbacks: PoolCallbacks[T],
        config: PoolConfig | None = None,
    ):
        self.keys = keys
        self.callbacks = callbacks
        self.config = config or PoolConfig()

        self._items: dict[str, PoolItem[T]] = {
            key: PoolItem(key=key) for key in keys
        }
        self._available: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._health_task: asyncio.Task[None] | None = None

    async def start(self, wait_for_one: bool = True) -> None:
        """Start the pool: begin creating items and health checking.

        Args:
            wait_for_one: If True, wait until at least one item is ready.
        """
        tasks = [asyncio.create_task(self._create_item(key)) for key in self.keys]

        if self.callbacks.check:
            self._health_task = asyncio.create_task(self._health_loop())

        if wait_for_one:
            while self._available.empty():
                if all(t.done() for t in tasks) and self._available.empty():
                    raise RuntimeError("All items failed to create")
                await asyncio.sleep(0.5)

    async def _create_item(self, key: str) -> None:
        """Create a single item with retry logic."""
        item = self._items[key]

        while item.create_attempts < self.config.max_create_attempts:
            item.status = ItemStatus.CREATING
            item.create_attempts += 1

            if self.callbacks.on_creating:
                self.callbacks.on_creating(key)

            try:
                value = await self.callbacks.create(key)
                item.value = value
                item.status = ItemStatus.READY
                item.create_attempts = 0

                if self.callbacks.on_ready:
                    self.callbacks.on_ready(key)

                await self._available.put(key)
                return

            except Exception as e:
                logger.error("Failed to create %s (attempt %d/%d): %s", key, item.create_attempts, self.config.max_create_attempts, e)

                if self.callbacks.on_unhealthy:
                    self.callbacks.on_unhealthy(key, str(e))

                if item.create_attempts < self.config.max_create_attempts:
                    await asyncio.sleep(self.config.create_retry_delay)
                else:
                    item.status = ItemStatus.UNHEALTHY

    async def _health_loop(self) -> None:
        """Background loop to check health of ready/in-use items."""
        while not self._shutdown:
            await asyncio.sleep(self.config.health_check_interval)

            # Only check items that were working (not already unhealthy)
            for key, item in self._items.items():
                if self._shutdown:
                    break

                # Proactively check healthy items (detect preemption early)
                if item.status == ItemStatus.READY and item.value is not None:
                    if self.callbacks.check:
                        try:
                            healthy = await self.callbacks.check(key, item.value)
                            if not healthy:
                                await self._mark_unhealthy(key, "health check failed")
                        except Exception as e:
                            await self._mark_unhealthy(key, str(e))

    async def acquire(self) -> tuple[str, T]:
        """Acquire an item from the pool.

        Returns (key, value) tuple. Blocks until an item is available.
        """
        while True:
            key = await self._available.get()
            item = self._items[key]

            if item.status != ItemStatus.READY or item.value is None:
                continue

            item.status = ItemStatus.IN_USE
            return key, item.value

    async def release(self, key: str, healthy: bool = True) -> None:
        """Release an item back to the pool.

        Args:
            key: The item key
            healthy: If False, item will be marked unhealthy and recreated
        """
        item = self._items.get(key)
        if item is None:
            return

        if healthy and item.value is not None:
            item.status = ItemStatus.READY
            await self._available.put(key)
        else:
            await self._mark_unhealthy(key, "marked unhealthy on release")

    async def _mark_unhealthy(self, key: str, reason: str) -> None:
        """Mark an item as unhealthy and trigger recovery."""
        item = self._items.get(key)
        if item is None:
            return

        item.status = ItemStatus.UNHEALTHY
        if self.callbacks.on_unhealthy:
            self.callbacks.on_unhealthy(key, reason)

        if item.create_attempts < self.config.max_create_attempts:
            asyncio.create_task(self._recover_item(key))

    async def _recover_item(self, key: str) -> None:
        """Recover an unhealthy item."""
        item = self._items.get(key)
        if item is None:
            return

        if item.value is not None:
            try:
                await self.callbacks.destroy(key, item.value)
            except Exception:
                pass
            item.value = None

        item.create_attempts = 0

        await asyncio.sleep(self.config.create_retry_delay)
        await self._create_item(key)

    async def close(self) -> None:
        """Shutdown the pool and destroy all items."""
        self._shutdown = True

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        async def destroy_one(key: str, item: PoolItem[T]) -> None:
            if self.callbacks.on_stopping:
                self.callbacks.on_stopping(key)

            try:
                await self.callbacks.destroy(key, item.value)
            except Exception:
                pass

            item.status = ItemStatus.DESTROYED
            if self.callbacks.on_destroyed:
                self.callbacks.on_destroyed(key)

        await asyncio.gather(*[
            destroy_one(key, item) for key, item in self._items.items()
        ])

    @property
    def ready_count(self) -> int:
        """Number of ready items."""
        return sum(1 for i in self._items.values() if i.status == ItemStatus.READY)

    @property
    def unhealthy_count(self) -> int:
        """Number of unhealthy items."""
        return sum(1 for i in self._items.values() if i.status == ItemStatus.UNHEALTHY)

    def status(self) -> dict[str, ItemStatus]:
        """Get status of all items."""
        return {key: item.status for key, item in self._items.items()}
