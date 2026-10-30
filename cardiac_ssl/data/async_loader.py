"""Asynchronous multiprocess data loading for FM-Bench.

Uses the same batchgenerators architecture as nnSSL:
workers continuously generate batches ahead of the training process while
the wrapper exposes a fixed number of iterations per epoch.
"""

from __future__ import annotations

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)


class LimitedLenWrapper(NonDetMultiThreadedAugmenter):
    """Asynchronous augmenter with a fixed apparent epoch length."""

    def __init__(self, length: int, *args, **kwargs):
        if length <= 0:
            raise ValueError("length must be > 0")

        super().__init__(*args, **kwargs)
        self.length = int(length)
        self._fmbench_closed = False

    def __len__(self) -> int:
        return self.length

    def shutdown(self) -> None:
        """Stop background workers and result threads safely.

        batchgenerators owns the worker processes/queues. Calling
        ``_finish`` before interpreter shutdown prevents those workers
        from being torn down while their result thread is still active.
        """
        if self._fmbench_closed:
            return

        self._fmbench_closed = True

        finish = getattr(self, "_finish", None)

        if callable(finish):
            finish()

    def close(self) -> None:
        """Alias used by FM-Bench training cleanup."""
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False

    def __del__(self):
        # Interpreter shutdown may already have partially destroyed
        # multiprocessing state, so destructor cleanup must never raise.
        try:
            self.shutdown()
        except Exception:
            pass
