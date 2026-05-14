"""Benchmark runner — executes one adapter against the loaded dataset."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

from raucle_bench.adapter import Adapter, Prediction
from raucle_bench.dataset import Prompt


@dataclass
class RunRecord:
    """One prompt × one adapter prediction, plus latency."""

    prompt_id: str
    attack_class: str
    category: str
    expected_action: str
    predicted_action: str
    confidence: float
    detail: str
    latency_ms: float
    error: str = ""


@dataclass
class RunResult:
    """Full output of running one adapter against the dataset."""

    adapter_name: str
    adapter_version: str
    records: list[RunRecord] = field(default_factory=list)
    setup_failed: str = ""
    """Non-empty if the adapter's setup() raised; the run was skipped."""


def run_adapter(adapter: Adapter, prompts: list[Prompt]) -> RunResult:
    """Run *adapter* over every prompt. Returns one :class:`RunResult`.

    Per-prompt exceptions are caught and recorded as ``error`` on the
    individual record. Setup-time exceptions abort the run for this
    adapter but leave the rest of the benchmark unaffected.
    """
    result = RunResult(adapter_name=adapter.name, adapter_version=adapter.version)

    try:
        adapter.setup()
    except Exception as exc:
        result.setup_failed = str(exc)
        return result

    try:
        for prompt in prompts:
            t0 = time.perf_counter()
            try:
                prediction = adapter.predict(prompt.prompt)
            except Exception as exc:
                prediction = Prediction(action="ALLOW", error=str(exc))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            result.records.append(
                RunRecord(
                    prompt_id=prompt.id,
                    attack_class=prompt.attack_class,
                    category=prompt.category,
                    expected_action=prompt.expected_action,
                    predicted_action=prediction.action,
                    confidence=prediction.confidence,
                    detail=prediction.detail,
                    latency_ms=latency_ms,
                    error=prediction.error,
                )
            )
    finally:
        with contextlib.suppress(Exception):
            adapter.teardown()

    return result
