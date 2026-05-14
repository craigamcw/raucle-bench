# Raucle-bench dataset

Curated prompts for benchmarking prompt-injection detection. Every record is one JSONL line in a per-category file.

## Schema

```json
{
  "id": "INJ-001",
  "prompt": "...",
  "category": "instruction_override",
  "attack_class": "direct_injection",
  "expected_action": "BLOCK",
  "severity": "HIGH",
  "source": "raucle-bench-curated",
  "notes": "Optional context"
}
```

Field reference:

| Field | Required | Description |
|---|---|---|
| `id` | yes | Stable identifier. Format: `<CLASS>-<NNN>`. Never re-use across versions. |
| `prompt` | yes | The test prompt. ASCII unless the test exercises Unicode. |
| `category` | yes | Specific technique tested (e.g. `instruction_override`, `dan_jailbreak`). |
| `attack_class` | yes | One of: `direct_injection`, `jailbreak`, `data_exfiltration`, `tool_abuse`, `evasion`, `benign`. |
| `expected_action` | yes | What a correctly-configured detector should produce: `BLOCK`, `ALERT`, `ALLOW`. |
| `severity` | yes | `CRITICAL` / `HIGH` / `MEDIUM` / `LOW` for attacks; `NONE` for benign. |
| `source` | yes | Origin label. See sources below. |
| `notes` | no | Free-form context. |

## Sources

- `raucle-bench-curated` — original or substantially-rephrased examples written for this benchmark. Licensed under the dataset MIT licence.
- `public-domain-paraphrase` — paraphrased from documented attack patterns in academic / industry literature.
- `synthetic` — programmatically generated patterns (e.g. encoding variants).

We deliberately do not redistribute datasets that carry restrictive licenses. If you contribute a record, please ensure you have the right to share it under MIT.

## Adding a category

To add a new attack class:

1. Create `datasets/<class>.jsonl`.
2. Add 30+ records covering distinct techniques within the class.
3. Submit a PR. The CI checks IDs are unique and schema fields are present.

## Adding a record

1. Open an issue describing the technique you want represented (optional but recommended for novel classes).
2. Add the JSONL line to the appropriate file. Use the next free `id` in the sequence.
3. Verify `python scripts/validate_dataset.py` passes.
4. Open a PR with the `dataset` label.

## Ethical considerations

These prompts are illustrative of attack *patterns* used in the wild against LLM systems. They are intended for defensive security research. Do not redistribute the dataset without making the intended use clear.

Some prompts intentionally resemble known jailbreaks because that is exactly what the dataset must test against. The dataset does not contain operational instructions for causing harm — only the meta-patterns detectors must recognise.
