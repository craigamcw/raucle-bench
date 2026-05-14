"""Dataset loader + validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from raucle_bench.dataset import (
    VALID_ACTIONS,
    VALID_ATTACK_CLASSES,
    Prompt,
    load_dataset,
    summarise,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"


class TestProductionDataset:
    """Validates the dataset that ships with this repo."""

    def test_loads_without_error(self):
        prompts = load_dataset(DATASETS_DIR)
        assert len(prompts) >= 100, "dataset should have at least 100 prompts"

    def test_every_id_is_unique(self):
        prompts = load_dataset(DATASETS_DIR)
        ids = [p.id for p in prompts]
        assert len(ids) == len(set(ids))

    def test_every_attack_class_represented(self):
        prompts = load_dataset(DATASETS_DIR)
        classes = {p.attack_class for p in prompts}
        for required in (
            "direct_injection",
            "jailbreak",
            "data_exfiltration",
            "tool_abuse",
            "evasion",
            "indirect_injection",
            "benign",
        ):
            assert required in classes, f"missing class {required}"

    def test_has_meaningful_benign_baseline(self):
        prompts = load_dataset(DATASETS_DIR)
        benign = [p for p in prompts if p.attack_class == "benign"]
        assert len(benign) >= 30, "need ≥30 benign prompts for FPR signal"

    def test_summary_shape(self):
        prompts = load_dataset(DATASETS_DIR)
        s = summarise(prompts)
        assert s["total"] == len(prompts)
        assert "benign" in s["by_attack_class"]


class TestSchemaEnforcement:
    """The validator must reject malformed records."""

    def _write_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "x.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records))
        return path

    def test_rejects_missing_required_field(self, tmp_path: Path):
        self._write_jsonl(tmp_path, [{"id": "X-1", "prompt": "hi"}])
        with pytest.raises(ValueError, match="missing fields"):
            load_dataset(tmp_path)

    def test_rejects_unknown_attack_class(self, tmp_path: Path):
        self._write_jsonl(
            tmp_path,
            [
                {
                    "id": "X-1",
                    "prompt": "hi",
                    "category": "x",
                    "attack_class": "nonsense",
                    "expected_action": "BLOCK",
                    "severity": "HIGH",
                    "source": "test",
                }
            ],
        )
        with pytest.raises(ValueError, match="attack_class"):
            load_dataset(tmp_path)

    def test_rejects_unknown_action(self, tmp_path: Path):
        self._write_jsonl(
            tmp_path,
            [
                {
                    "id": "X-1",
                    "prompt": "hi",
                    "category": "x",
                    "attack_class": "benign",
                    "expected_action": "BANANA",
                    "severity": "NONE",
                    "source": "test",
                }
            ],
        )
        with pytest.raises(ValueError, match="expected_action"):
            load_dataset(tmp_path)

    def test_rejects_inconsistent_benign(self, tmp_path: Path):
        self._write_jsonl(
            tmp_path,
            [
                {
                    "id": "X-1",
                    "prompt": "hi",
                    "category": "x",
                    "attack_class": "benign",
                    "expected_action": "BLOCK",
                    "severity": "HIGH",
                    "source": "test",
                }
            ],
        )
        with pytest.raises(ValueError, match="benign.*ALLOW"):
            load_dataset(tmp_path)

    def test_rejects_duplicate_ids(self, tmp_path: Path):
        rec = {
            "id": "X-1",
            "prompt": "hi",
            "category": "x",
            "attack_class": "benign",
            "expected_action": "ALLOW",
            "severity": "NONE",
            "source": "test",
        }
        self._write_jsonl(tmp_path, [rec, rec])
        with pytest.raises(ValueError, match="duplicate id"):
            load_dataset(tmp_path)


class TestPromptDataclass:
    def test_is_malicious(self):
        attack = Prompt(
            id="A",
            prompt="hi",
            category="x",
            attack_class="jailbreak",
            expected_action="BLOCK",
            severity="HIGH",
            source="test",
        )
        benign = Prompt(
            id="B",
            prompt="hi",
            category="x",
            attack_class="benign",
            expected_action="ALLOW",
            severity="NONE",
            source="test",
        )
        assert attack.is_malicious
        assert not benign.is_malicious


class TestConstants:
    def test_action_set(self):
        assert {"ALLOW", "ALERT", "BLOCK"} == VALID_ACTIONS

    def test_attack_class_set_includes_benign(self):
        assert "benign" in VALID_ATTACK_CLASSES
