"""Raucle-bench — public adversarial leaderboard for prompt-injection detection."""

__version__ = "0.1.0"
__author__ = "Raucle"
__license__ = "MIT"

from raucle_bench.adapter import Adapter, Prediction
from raucle_bench.dataset import Prompt, load_dataset, summarise
from raucle_bench.report import build_report, render_html, render_markdown, write_json
from raucle_bench.runner import RunRecord, RunResult, run_adapter
from raucle_bench.scoring import ClassScore, Score, score

__all__ = [
    "Adapter",
    "Prediction",
    "Prompt",
    "load_dataset",
    "summarise",
    "RunRecord",
    "RunResult",
    "run_adapter",
    "Score",
    "ClassScore",
    "score",
    "build_report",
    "render_html",
    "render_markdown",
    "write_json",
    "__version__",
]
