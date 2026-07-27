from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark.reviewer_eval.datasets import (
    JAILBOUND_ATTACK_TYPES,
    JAILBOUND_RISK_CATEGORIES,
    SourceDataError,
    load_source,
    load_source_with_report,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATHS = {
    "advbench": ROOT / "01benchmark-datasets/downloaded_datasets/AdvBench/data/advbench/harmful_behaviors.csv",
    "harmbench": ROOT / "01benchmark-datasets/downloaded_datasets/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    "safetybench": ROOT / "01benchmark-datasets/downloaded_datasets/SafetyBench/opensource_data/test_en.json",
    "sg_bench": ROOT / "01benchmark-datasets/downloaded_datasets/SG-Bench/datasets/SG-Bench/original_query/malicious_instruction.json",
    "jailbreakbench": ROOT / "data/sft/jailbreakbench/data/harmful-behaviors.csv",
    "jailbound": ROOT / "data/sft/attack_sft_v2.json",
    "s_eval": ROOT / "01benchmark-datasets/S-Eval_attack_en_full.jsonl",
}
HARMBENCH_TARGETS = ROOT / "01benchmark-datasets/downloaded_datasets/HarmBench/data/optimizer_targets/harmbench_targets_text.json"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def adapter_paths(tmp_path: Path) -> dict[str, Path]:
    advbench = tmp_path / "advbench.csv"
    _write_csv(advbench, ["goal", "target"], [{"goal": "Goal", "target": "Target"}])
    harmbench = tmp_path / "harmbench.csv"
    _write_csv(harmbench, ["Behavior", "BehaviorID", "ContextString", "FunctionalCategory", "SemanticCategory"], [{"Behavior": "Goal", "BehaviorID": "B1", "ContextString": "Context", "FunctionalCategory": "standard", "SemanticCategory": "illegal"}, {"Behavior": "Excluded", "BehaviorID": "B2", "ContextString": "", "FunctionalCategory": "copyright", "SemanticCategory": "copyright"}])
    (tmp_path / "targets.json").write_text(json.dumps({"B1": "Target", "B2": "Ignored"}), encoding="utf-8")
    safetybench = tmp_path / "safetybench.json"
    safetybench.write_text(json.dumps([{"question": "Question", "options": ["One", "Two"], "category": "Illegal Activities"}]), encoding="utf-8")
    sg_bench = tmp_path / "sg_bench.json"
    sg_bench.write_text(json.dumps([{"query": "Query", "reference_responses": ["Reference"], "safety_type": "[O5: label]"}]), encoding="utf-8")
    jailbreakbench = tmp_path / "jailbreakbench.csv"
    _write_csv(jailbreakbench, ["Goal", "Target", "Category"], [{"Goal": "Goal", "Target": "Target", "Category": "Privacy"}])
    jailbound = tmp_path / "jailbound.json"
    jailbound.write_text(json.dumps([{"instruction": "Attack Type: Persuasion & Deception, Risk Category: Illegal Wrongdoing & Criminal Enablement, and Attack Domain: Education, generate training data.", "input": "Intent", "output": "Attack"}]), encoding="utf-8")
    s_eval = tmp_path / "S-Eval_attack_en_full.jsonl"
    s_eval.write_text(json.dumps({"prompt": "Prompt", "risk_type": "Crimes_and_Illegal_Activities", "ext": json.dumps({"category": "positive_induction"})}) + "\n", encoding="utf-8")
    return {"advbench": advbench, "harmbench": harmbench, "harmbench_targets": tmp_path / "targets.json", "safetybench": safetybench, "sg_bench": sg_bench, "jailbreakbench": jailbreakbench, "jailbound": jailbound, "s_eval": s_eval}


@pytest.mark.parametrize(("source", "expected"), [("advbench", 520), ("harmbench", 320), ("safetybench", 11435), ("sg_bench", 1442), ("jailbreakbench", 100), ("jailbound", 10000), ("s_eval", 100000)])
def test_real_source_counts(source: str, expected: int) -> None:
    records = load_source(source, DATA_PATHS[source], harmbench_targets_path=HARMBENCH_TARGETS)
    if source == "harmbench":
        assert len(records) == 240
    else:
        assert len(records) == expected


def test_real_harmbench_report_records_the_fixed_exclusion() -> None:
    records, report = load_source_with_report(
        "harmbench",
        DATA_PATHS["harmbench"],
        harmbench_targets_path=HARMBENCH_TARGETS,
    )

    assert report.raw_count == 320
    assert report.eligible_count == 240
    assert report.exclusions == {"copyright": 80}
    assert len(records) == 240


def test_harmbench_records_the_configured_copyright_exclusion(adapter_paths: dict[str, Path]) -> None:
    records, report = load_source_with_report("harmbench", adapter_paths["harmbench"], harmbench_targets_path=adapter_paths["harmbench_targets"])
    assert len(records) == 1
    assert report.raw_count == 2
    assert report.eligible_count == 1
    assert report.exclusions == {"copyright": 1}
    assert records[0].attack_text == "Context\n\n---\n\nGoal"


def test_jailbound_adapter_separates_intent_and_attack(adapter_paths: dict[str, Path]) -> None:
    record = load_source("jailbound", adapter_paths["jailbound"])[0]
    assert record.intent and record.attack_text and record.intent != record.attack_text
    assert record.source_attack_label in JAILBOUND_ATTACK_TYPES
    assert record.source_risk_label in JAILBOUND_RISK_CATEGORIES


def test_s_eval_adapter_parses_ext_json(adapter_paths: dict[str, Path]) -> None:
    record = load_source("s_eval", adapter_paths["s_eval"])[0]
    assert record.source_risk_label == "Crimes_and_Illegal_Activities"
    assert record.source_attack_label == "positive_induction"
    assert record.language == "en"


def test_unlabeled_sources_keep_direct_request_control(adapter_paths: dict[str, Path]) -> None:
    for source in ("advbench", "harmbench", "safetybench", "sg_bench", "jailbreakbench"):
        kwargs = {"harmbench_targets_path": adapter_paths["harmbench_targets"]} if source == "harmbench" else {}
        assert load_source(source, adapter_paths[source], **kwargs)[0].source_attack_label == "direct_request"


def test_source_row_id_includes_source_row_and_normalized_attack_text(
    adapter_paths: dict[str, Path],
) -> None:
    record = load_source("advbench", adapter_paths["advbench"])[0]

    assert record.source_row_id.startswith("advbench:000000:")
    assert record.source_row_id != record.__class__(
        **{**record.__dict__, "source_row": 1}
    ).source_row_id
    assert record.source_row_id != record.__class__(
        **{**record.__dict__, "attack_text": "Changed"}
    ).source_row_id


def test_sg_bench_keeps_rows_with_no_reference_response(adapter_paths: dict[str, Path]) -> None:
    adapter_paths["sg_bench"].write_text(json.dumps([{"query": "Query", "reference_responses": [], "safety_type": "[O5: label]"}]), encoding="utf-8")
    assert load_source("sg_bench", adapter_paths["sg_bench"])[0].target_text is None


def test_sg_bench_keeps_rows_with_null_reference_response(adapter_paths: dict[str, Path]) -> None:
    adapter_paths["sg_bench"].write_text(json.dumps([{"query": "Query", "reference_responses": None, "safety_type": "[O5: label]"}]), encoding="utf-8")
    assert load_source("sg_bench", adapter_paths["sg_bench"])[0].target_text is None


def test_jailbound_requires_all_native_labels(adapter_paths: dict[str, Path]) -> None:
    adapter_paths["jailbound"].write_text(json.dumps([{"instruction": "Attack Type: Persuasion & Deception", "input": "Intent", "output": "Attack"}]), encoding="utf-8")
    with pytest.raises(SourceDataError, match="JailBound row 0"):
        load_source("jailbound", adapter_paths["jailbound"])


def test_s_eval_requires_english_source_file(adapter_paths: dict[str, Path]) -> None:
    path = adapter_paths["s_eval"].with_name("S-Eval_attack_zh_full.jsonl")
    adapter_paths["s_eval"].replace(path)
    with pytest.raises(SourceDataError, match="English"):
        load_source("s_eval", path)


def test_adapter_rejects_missing_required_csv_field(adapter_paths: dict[str, Path]) -> None:
    _write_csv(adapter_paths["advbench"], ["target"], [{"target": "Target"}])

    with pytest.raises(SourceDataError, match="text value"):
        load_source("advbench", adapter_paths["advbench"])


def test_adapter_rejects_non_object_json_row(adapter_paths: dict[str, Path]) -> None:
    adapter_paths["safetybench"].write_text(json.dumps(["not-an-object"]), encoding="utf-8")

    with pytest.raises(SourceDataError, match="must be an object"):
        load_source("safetybench", adapter_paths["safetybench"])
