"""Adapter contract — every detector wraps itself in this interface.

To benchmark a new tool, write a class that implements :class:`Adapter`.
Register it in :data:`ADAPTERS` (or expose it on the CLI). The runner does
not care about the underlying tool — it only needs a stable
``predict(prompt) -> Prediction`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Allowed values: "ALLOW" | "ALERT" | "BLOCK".
Action = str


@dataclass(frozen=True)
class Prediction:
    """A single prediction emitted by an adapter."""

    action: Action
    """Adapter's recommended action: ``ALLOW`` / ``ALERT`` / ``BLOCK``."""

    confidence: float = 0.0
    """Adapter's own confidence score (0.0 – 1.0). Optional."""

    detail: str = ""
    """Optional human-readable explanation (matched rule, category, etc.)."""

    error: str = ""
    """Set when the adapter threw an exception. The runner records this and
    moves on rather than aborting the entire run."""

    @property
    def is_error(self) -> bool:
        return bool(self.error)


class Adapter(Protocol):
    """Protocol every benchmark adapter must satisfy."""

    name: str
    """Stable identifier shown on the leaderboard (e.g. ``raucle-detect-0.5.0``)."""

    version: str
    """The adapted tool's version. Recorded with every result row."""

    def predict(self, prompt: str) -> Prediction:
        """Return a :class:`Prediction` for *prompt*. MUST NOT raise."""
        ...

    def setup(self) -> None:
        """Optional one-time initialisation. Called once before the run."""

    def teardown(self) -> None:
        """Optional cleanup. Called once after the run."""
