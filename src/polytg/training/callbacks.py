"""Small reusable training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_value: float = float("inf")
    epochs_without_improvement: int = 0

    def update(self, value: float) -> bool:
        """Update state and return whether training should stop."""
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience
