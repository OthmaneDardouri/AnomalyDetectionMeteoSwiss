"""Shared early-stopping helper."""

from typing import Optional

from anom_detect.logging_utils import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop after ``patience`` epochs without a ``delta`` improvement.

    Feed it one metric consistently: mixing two series corrupts the counter,
    since an improvement in one resets the count for the other.
    """

    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = False) -> None:
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss: Optional[float] = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss: float) -> None:
        """Record ``val_loss`` for this epoch and update ``stop_training``."""
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    logger.info("Stopping early as no improvement has been observed.")
