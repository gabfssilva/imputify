"""Thread-safe iterator and callbacks for streaming training/imputation progress."""

import queue
from dataclasses import dataclass

from transformers import TrainerCallback


@dataclass
class TrainingStep:
    """Snapshot of training state emitted after each logging step."""

    step: int
    epoch: float
    train_loss: float
    eval_loss: float
    grad_norm: float
    learning_rate: float


@dataclass
class ImputationStep:
    """Progress update emitted after each cell imputation attempt."""
    rows_completed: int
    rows_total: int
    cells_imputed: int
    cells_total: int
    retries: int = 0
    failures: int = 0

    @property
    def progress(self) -> float:
        """Fraction of cells imputed (0.0 to 1.0)."""
        return self.cells_imputed / self.cells_total if self.cells_total > 0 else 1.0

class PushableIterator:
    """Thread-safe iterator backed by a queue.

    Allows a producer thread to push items and a consumer to
    iterate. Calling close() signals end of iteration.
    """

    _SENTINEL = object()

    def __init__(self):
        self._q = queue.Queue()
        self._closed = False

    def push(self, item):
        if self._closed:
            raise RuntimeError("Cannot push to closed iterator")
        self._q.put(item)

    def close(self):
        if not self._closed:
            self._closed = True
            self._q.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item

class PushableTrainingCallback(TrainerCallback):
    """HuggingFace TrainerCallback that pushes TrainingSteps to a PushableIterator."""

    def __init__(self, stream: PushableIterator):
        self.stream = stream
        self._last_epoch_train_loss: float | None = None
        self._last_epoch_eval_loss: float | None = None
        self._last_epoch_grad_norm: float | None = None

    def on_epoch_end(self, args, state, control, **kwargs):
        log = state.log_history[-1] if state.log_history else {}
        self._last_epoch_train_loss = log.get("loss") or log.get("train_loss")
        self._last_epoch_eval_loss = log.get("eval_loss")
        self._last_epoch_grad_norm = log.get("grad_norm")

    def on_log(self, args, state, control, **kwargs):
        log = state.log_history[-1] if state.log_history else {}

        step = state.global_step
        epoch = state.epoch
        learning_rate = log.get("learning_rate")

        self.stream.push(TrainingStep(
            step=step,
            epoch=epoch,
            train_loss=self._last_epoch_train_loss,
            eval_loss=self._last_epoch_eval_loss,
            grad_norm=self._last_epoch_grad_norm,
            learning_rate=learning_rate,
        ))
