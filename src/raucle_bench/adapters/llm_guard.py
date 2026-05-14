"""Adapter for the open-source LLM Guard library.

LLM Guard is one of the most-cited OSS competitors to raucle-detect. Its
``PromptInjection`` input scanner uses a transformer model and returns a
risk score plus a validity bool. We map that to our ``Action`` taxonomy:

- score = 0 → ALLOW
- 0 < score < threshold → ALERT
- score ≥ threshold → BLOCK

LLM Guard's ``PromptInjection`` scanner has a default threshold of 0.5;
we expose both ``threshold`` and an internal alert tier to keep the
mapping honest.

Install with::

    pip install 'raucle-bench[llm-guard]'

The ``llm-guard`` package pulls ~1 GB of model artifacts on first use,
which is why this adapter is opt-in and shipped behind an extra.
"""

from __future__ import annotations

from raucle_bench.adapter import Prediction


class LLMGuardAdapter:
    """Wraps ``llm_guard.input_scanners.PromptInjection``."""

    def __init__(self, *, threshold: float = 0.5, alert_threshold: float = 0.3) -> None:
        if not 0.0 <= alert_threshold <= threshold <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 <= alert ({alert_threshold}) "
                f"<= block ({threshold}) <= 1"
            )
        self.threshold = threshold
        self.alert_threshold = alert_threshold
        self._scanner = None
        # Late import — keep the adapter file importable even when the
        # llm-guard package is not installed. The runner will mark
        # setup_failed gracefully.
        try:
            import llm_guard  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "llm-guard is required for LLMGuardAdapter. "
                "Install with: pip install 'raucle-bench[llm-guard]'"
            ) from exc

    @property
    def name(self) -> str:
        return f"llm-guard-prompt-injection-t{self.threshold:g}"

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("llm-guard")
        except Exception:
            return "unknown"

    def setup(self) -> None:
        from llm_guard.input_scanners import PromptInjection  # type: ignore[import-untyped]

        # PromptInjection downloads its model on first instantiation.
        self._scanner = PromptInjection(threshold=self.threshold)

    def teardown(self) -> None:
        self._scanner = None

    def predict(self, prompt: str) -> Prediction:
        if self._scanner is None:
            return Prediction(action="ALLOW", error="adapter not set up")
        try:
            # PromptInjection.scan returns (sanitised_prompt, is_valid, risk_score).
            _sanitised, is_valid, risk_score = self._scanner.scan(prompt)
        except Exception as exc:
            return Prediction(action="ALLOW", error=str(exc))

        # is_valid==False means LLM Guard recommended blocking.
        if not is_valid or risk_score >= self.threshold:
            action = "BLOCK"
        elif risk_score >= self.alert_threshold:
            action = "ALERT"
        else:
            action = "ALLOW"

        return Prediction(
            action=action,
            confidence=float(risk_score),
            detail=f"risk_score={risk_score:.3f}",
        )
