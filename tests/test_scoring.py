"""Scoring methodology tests — every claim in the leaderboard rides on these."""

from __future__ import annotations

import pytest

from raucle_bench.runner import RunRecord, RunResult
from raucle_bench.scoring import score


def _record(
    pid: str,
    is_attack: bool,
    predicted: str,
    *,
    latency_ms: float = 1.0,
    expected: str | None = None,
) -> RunRecord:
    return RunRecord(
        prompt_id=pid,
        attack_class="jailbreak" if is_attack else "benign",
        category="t",
        expected_action=expected or ("BLOCK" if is_attack else "ALLOW"),
        predicted_action=predicted,
        confidence=0.5,
        detail="",
        latency_ms=latency_ms,
    )


def _result(records: list[RunRecord]) -> RunResult:
    return RunResult(adapter_name="t", adapter_version="0", records=records)


class TestPerfectDetector:
    def test_scores_at_ceiling(self):
        recs = [_record("a", True, "BLOCK"), _record("b", False, "ALLOW")]
        s = score(_result(recs))
        assert s.precision == 1.0
        assert s.recall == 1.0
        assert s.f1 == 1.0
        assert s.false_positive_rate == 0.0
        assert s.strict_match_rate == 1.0


class TestAlwaysAllow:
    def test_zero_recall_zero_fpr(self):
        recs = [_record("a", True, "ALLOW"), _record("b", False, "ALLOW")]
        s = score(_result(recs))
        assert s.recall == 0.0
        assert s.precision == 0.0
        assert s.false_positive_rate == 0.0
        assert s.detection_rate == 0.0


class TestAlwaysBlock:
    def test_full_recall_max_fpr(self):
        recs = [_record("a", True, "BLOCK"), _record("b", False, "BLOCK")]
        s = score(_result(recs))
        assert s.recall == 1.0
        assert s.false_positive_rate == 1.0
        # precision = 1/2 because half the BLOCKs were on benign
        assert s.precision == 0.5


class TestMixed:
    def test_realistic_scenario(self):
        # 4 attacks, 3 benigns. Detector catches 3 attacks, false-positives once.
        recs = [
            _record("a1", True, "BLOCK"),
            _record("a2", True, "ALERT"),
            _record("a3", True, "BLOCK"),
            _record("a4", True, "ALLOW"),  # missed
            _record("b1", False, "ALLOW"),
            _record("b2", False, "ALLOW"),
            _record("b3", False, "ALERT"),  # false positive
        ]
        s = score(_result(recs))
        assert s.true_positive == 3
        assert s.false_negative == 1
        assert s.false_positive == 1
        assert s.true_negative == 2
        assert s.precision == 0.75
        assert s.recall == 0.75
        assert s.false_positive_rate == pytest.approx(1 / 3, abs=1e-6)
        assert s.f1 == pytest.approx(0.75, abs=1e-6)


class TestStrictMatch:
    def test_alert_counts_as_detection_but_not_strict_match_on_block_expected(self):
        # Expected BLOCK, predicted ALERT — caught the attack but recommended
        # a weaker action. detection_rate=1.0 but strict_match=0.
        rec = _record("a", True, "ALERT", expected="BLOCK")
        s = score(_result([rec]))
        assert s.detection_rate == 1.0
        assert s.strict_match_rate == 0.0


class TestLatencyPercentiles:
    def test_p50_and_p99_computed(self):
        recs = [_record(f"a{i}", True, "BLOCK", latency_ms=float(i)) for i in range(100)]
        s = score(_result(recs))
        # 100 values 0..99; p50 ≈ 49.5, p99 ≈ 98.01
        assert 48 <= s.latency_p50_ms <= 51
        assert 97 <= s.latency_p99_ms <= 99


class TestPerClassBreakdown:
    def test_classes_aggregate_independently(self):
        recs = [
            _record("a1", True, "BLOCK"),
            _record("a2", True, "BLOCK"),
            _record("b1", False, "ALLOW"),
        ]
        s = score(_result(recs))
        classes = {c.attack_class: c for c in s.per_class}
        assert classes["jailbreak"].detected == 2
        assert classes["benign"].detected == 0  # zero false positives


class TestSetupFailedShortCircuits:
    def test_setup_failure_produces_zero_score(self):
        result = RunResult(
            adapter_name="x",
            adapter_version="0",
            records=[],
            setup_failed="missing dep",
        )
        s = score(result)
        assert s.setup_failed == "missing dep"
        assert s.f1 == 0.0
