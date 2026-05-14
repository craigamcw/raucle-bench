"""Report-rendering tests."""

from __future__ import annotations

import json

from raucle_bench.dataset import Prompt
from raucle_bench.report import build_report, render_html, render_markdown
from raucle_bench.runner import RunRecord, RunResult
from raucle_bench.scoring import score


def _result(name: str, recs: list[RunRecord]) -> RunResult:
    return RunResult(adapter_name=name, adapter_version="0", records=recs)


def _rec(pid: str, attack: bool, predicted: str) -> RunRecord:
    return RunRecord(
        prompt_id=pid,
        attack_class="jailbreak" if attack else "benign",
        category="t",
        expected_action="BLOCK" if attack else "ALLOW",
        predicted_action=predicted,
        confidence=0.5,
        detail="",
        latency_ms=1.0,
    )


def _prompts() -> list[Prompt]:
    return [
        Prompt(
            id="a",
            prompt="x",
            category="t",
            attack_class="jailbreak",
            expected_action="BLOCK",
            severity="HIGH",
            source="t",
        )
    ]


class TestBuildReport:
    def test_leaderboard_sorted_by_f1_descending(self):
        good_recs = [_rec("a", True, "BLOCK"), _rec("b", False, "ALLOW")]
        bad_recs = [_rec("a", True, "ALLOW"), _rec("b", False, "ALLOW")]
        scores = [score(_result("bad", bad_recs)), score(_result("good", good_recs))]
        report = build_report(scores, _prompts(), run_id="2026-05-14T00-00-00Z")
        names = [row["adapter_name"] for row in report["leaderboard"]]
        assert names[0] == "good"
        assert names[1] == "bad"

    def test_includes_dataset_summary(self):
        s = score(_result("x", [_rec("a", True, "BLOCK")]))
        report = build_report([s], _prompts())
        assert report["dataset"]["total"] == 1


class TestMarkdownRendering:
    def test_has_header_and_table(self):
        s = score(_result("foo", [_rec("a", True, "BLOCK"), _rec("b", False, "ALLOW")]))
        md = render_markdown(build_report([s], _prompts(), run_id="2026-05-14"))
        assert "Raucle-bench leaderboard" in md
        assert "| Adapter " in md
        assert "`foo`" in md


class TestHTMLRendering:
    def test_produces_valid_looking_html(self):
        s = score(_result("foo", [_rec("a", True, "BLOCK"), _rec("b", False, "ALLOW")]))
        html_out = render_html(build_report([s], _prompts(), run_id="2026-05-14"))
        assert "<!DOCTYPE html>" in html_out
        assert "foo" in html_out
        assert "</table>" in html_out

    def test_setup_failed_adapter_renders_error_row(self):
        # Build a fake leaderboard row with setup_failed
        fake = {
            "run_id": "x",
            "schema_version": "v1",
            "dataset": {"total": 0, "by_attack_class": {}},
            "leaderboard": [
                {
                    "adapter_name": "broken",
                    "adapter_version": "0",
                    "setup_failed": "missing dep",
                }
            ],
        }
        html_out = render_html(fake)
        assert "missing dep" in html_out


class TestJsonRoundTrip:
    def test_serialisable(self, tmp_path):
        s = score(_result("foo", [_rec("a", True, "BLOCK")]))
        report = build_report([s], _prompts(), run_id="2026-05-14")
        path = tmp_path / "out.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        loaded = json.loads(path.read_text())
        assert loaded["run_id"] == "2026-05-14"
        assert loaded["leaderboard"][0]["adapter_name"] == "foo"
