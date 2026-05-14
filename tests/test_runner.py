"""Runner tests using fake adapters."""

from __future__ import annotations

from raucle_bench.adapter import Prediction
from raucle_bench.adapters.baseline import AlwaysAllow, AlwaysBlock
from raucle_bench.dataset import Prompt
from raucle_bench.runner import run_adapter


def _prompt(pid: str, is_attack: bool) -> Prompt:
    return Prompt(
        id=pid,
        prompt=f"prompt-{pid}",
        category="t",
        attack_class="jailbreak" if is_attack else "benign",
        expected_action="BLOCK" if is_attack else "ALLOW",
        severity="HIGH" if is_attack else "NONE",
        source="test",
    )


class TestBaselineAdapters:
    def test_always_allow_predicts_allow_for_every_prompt(self):
        prompts = [_prompt("a", True), _prompt("b", False)]
        result = run_adapter(AlwaysAllow(), prompts)
        assert len(result.records) == 2
        assert all(r.predicted_action == "ALLOW" for r in result.records)
        assert result.setup_failed == ""

    def test_always_block_predicts_block_for_every_prompt(self):
        prompts = [_prompt("a", True), _prompt("b", False)]
        result = run_adapter(AlwaysBlock(), prompts)
        assert all(r.predicted_action == "BLOCK" for r in result.records)

    def test_latency_is_recorded(self):
        prompts = [_prompt("a", True)]
        result = run_adapter(AlwaysAllow(), prompts)
        assert result.records[0].latency_ms >= 0.0


class _BrokenAdapter:
    name = "broken"
    version = "0"

    def setup(self):
        pass

    def teardown(self):
        pass

    def predict(self, prompt: str) -> Prediction:
        raise RuntimeError("kaboom")


class TestExceptionsHandled:
    def test_adapter_exception_recorded_not_raised(self):
        prompts = [_prompt("a", True), _prompt("b", False)]
        result = run_adapter(_BrokenAdapter(), prompts)
        assert len(result.records) == 2
        assert all(r.error == "kaboom" for r in result.records)
        # The adapter's default action when it raises is ALLOW so the
        # detection rate falls — but the run does not abort.
        assert all(r.predicted_action == "ALLOW" for r in result.records)


class _SetupFailsAdapter:
    name = "setup-fails"
    version = "0"

    def setup(self):
        raise RuntimeError("missing dep")

    def teardown(self):
        pass

    def predict(self, prompt: str) -> Prediction:
        return Prediction(action="ALLOW")


class TestSetupFailure:
    def test_setup_failure_short_circuits(self):
        prompts = [_prompt("a", True)]
        result = run_adapter(_SetupFailsAdapter(), prompts)
        assert result.records == []
        assert "missing dep" in result.setup_failed
