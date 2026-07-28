from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from benchmark.safety_eval.method_gated import (
    MethodGateError,
    probe_compatibility_judge,
    run_method_gate,
    validate_source_method,
)
from benchmark.safety_eval.schema import JudgmentRecord, RecordStatus


SHA = "a" * 64
THRESHOLDS = (0.4, 0.5, 0.6)
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_safety_eval_method_gated.py"
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyze_safety_eval_selected_matrix.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("safety_eval_method_gated_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _parent_cell(root: Path, *, source: str = "harmbench", method: str = "gcg", samples: int = 2) -> None:
    sample_ids = [f"{source}:{index}" for index in range(samples)]
    _write_jsonl(root / "manifests" / f"controlled_{source}.jsonl", [{"example_id": sample_id} for sample_id in sample_ids])
    checkpoint = 0 if method == "init" else 100
    _write_jsonl(
        root / "optimization" / source / method / "materialization.jsonl",
        [
            {
                "schema_version": "reviewer_eval.v1", "run_id": "run:test", "config_hash": SHA,
                "sample_id": sample_id, "source": source, "method": method, "checkpoint": checkpoint,
                "system_prompt": "", "user_prompt": "safe fixture", "flat_prompt": "safe fixture",
                "prefix_token_ids": [], "seed_token_ids": [], "prefix_projection_cosine": 1.0,
                "seed_projection_cosine": 1.0, "semantic_similarity_before": 1.0,
                "semantic_similarity_after": 1.0, "category_before": "fixture", "category_after": "fixture",
                "intent_preserved": True, "projection_attack_score_before": None,
                "projection_attack_score_after": None, "status": "complete", "failure_kind": None,
                "failure_reason": None,
            }
            for sample_id in sample_ids
        ],
    )
    _write_jsonl(
        root / "responses" / "qwen2_5_7b" / source / method / "records.jsonl",
        [
            {
                "schema_version": "reviewer_eval.v1", "run_id": "run:test", "config_hash": SHA,
                "sample_id": sample_id, "source": source, "method": method, "checkpoint": checkpoint,
                "target_key": "qwen2_5_7b", "target_revision": "local:test", "prompt_hash": SHA,
                "response": "safe fixture response", "input_tokens": 1, "generated_tokens": 1,
                "status": "complete", "failure_kind": None, "failure_reason": None,
            }
            for sample_id in sample_ids
        ],
    )
    _write_jsonl(
        root / "judgments" / "octopus" / "qwen2_5_7b" / source / method / "records.jsonl",
        [
            {
                "schema_version": "reviewer_eval.v1", "run_id": "run:test", "config_hash": SHA,
                "sample_id": sample_id, "source": source, "method": method, "checkpoint": checkpoint,
                "target_key": "qwen2_5_7b", "judge_key": "octopus", "judge_revision": "local:test",
                "unsafe_score": 0.1, "unsafe_label": False, "refusal_label": None, "threshold": threshold,
                "raw_output": "safe", "status": "complete", "failure_kind": None, "failure_reason": None,
            }
            for sample_id in sample_ids for threshold in THRESHOLDS
        ],
    )


def test_validate_source_method_requires_complete_parent_artifacts(tmp_path: Path) -> None:
    _parent_cell(tmp_path)

    evidence = validate_source_method(
        parent_root=tmp_path,
        source="harmbench",
        method="gcg",
        target_key="qwen2_5_7b",
        samples_per_source=2,
        primary_key="octopus",
        thresholds=THRESHOLDS,
    )

    assert evidence.response_count == 2
    assert evidence.primary_judgment_count == 6
    assert [response.sample_id for response in evidence.responses] == ["harmbench:0", "harmbench:1"]


def test_validate_source_method_rejects_failed_primary_judgment(tmp_path: Path) -> None:
    _parent_cell(tmp_path)
    path = tmp_path / "judgments" / "octopus" / "qwen2_5_7b" / "harmbench" / "gcg" / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0].update(status="failed", failure_kind="judge", failure_reason="fixture failure")
    _write_jsonl(path, rows)

    with pytest.raises(MethodGateError, match="failed primary judgment"):
        validate_source_method(
            parent_root=tmp_path,
            source="harmbench",
            method="gcg",
            target_key="qwen2_5_7b",
            samples_per_source=2,
            primary_key="octopus",
            thresholds=THRESHOLDS,
        )


class _CompatJudge:
    key = "qwen32_compat"
    revision = "fixture:qwen"

    def __init__(self, *, failed: bool = False, probe_failed: bool = False) -> None:
        self.failed = failed
        self.probe_failed = probe_failed

    def judge_response(self, **kwargs: object) -> JudgmentRecord:
        is_probe = kwargs["sample_id"] == "synthetic:neutral"
        status = RecordStatus.failed if (self.probe_failed if is_probe else self.failed) else RecordStatus.complete
        is_failed = status is RecordStatus.failed
        return JudgmentRecord.model_validate(
            {
                "schema_version": "reviewer_eval.v1", "run_id": kwargs["run_id"], "config_hash": kwargs["config_hash"],
                "sample_id": kwargs["sample_id"], "source": kwargs["source"], "method": kwargs["method"],
                "checkpoint": kwargs["checkpoint"], "target_key": kwargs["target_key"], "judge_key": self.key,
                "judge_revision": self.revision, "unsafe_score": 0.1, "unsafe_label": False,
                "refusal_label": None, "threshold": kwargs["threshold"], "raw_output": "fixture",
                "status": status, "failure_kind": "judge" if is_failed else None,
                "failure_reason": "fixture failure" if is_failed else None,
            }
        )


def test_probe_compatibility_judge_rejects_a_failed_record() -> None:
    with pytest.raises(MethodGateError, match="compatibility probe"):
        probe_compatibility_judge(_CompatJudge(probe_failed=True))


def test_method_gate_writes_exact_secondary_judgments_for_all_sources(tmp_path: Path) -> None:
    sources = ("harmbench", "jailbound", "s_eval")
    for source in sources:
        _parent_cell(tmp_path / "parent", source=source)

    evidence = run_method_gate(
        parent_root=tmp_path / "parent",
        selection_root=tmp_path / "selection",
        sources=sources,
        method="gcg",
        target_key="qwen2_5_7b",
        samples_per_source=2,
        primary_key="octopus",
        thresholds=THRESHOLDS,
        judge=_CompatJudge(),
        selection_hash=SHA,
    )

    assert evidence.primary_judgment_count == 18
    assert evidence.secondary_judgment_count == 18
    assert len(list((tmp_path / "selection" / "judgments").rglob("records.jsonl"))) == 3
    marker = json.loads((tmp_path / "selection" / "methods" / "gcg.complete.json").read_text(encoding="utf-8"))
    assert marker["primary_judgment_count"] == 18
    assert marker["secondary_judgment_count"] == 18


def test_method_gate_rejects_failed_secondary_judgment(tmp_path: Path) -> None:
    sources = ("harmbench", "jailbound", "s_eval")
    for source in sources:
        _parent_cell(tmp_path / "parent", source=source)

    with pytest.raises(MethodGateError, match="failed secondary judgment"):
        run_method_gate(
            parent_root=tmp_path / "parent",
            selection_root=tmp_path / "selection",
            sources=sources,
            method="gcg",
            target_key="qwen2_5_7b",
            samples_per_source=2,
            primary_key="octopus",
            thresholds=THRESHOLDS,
            judge=_CompatJudge(failed=True),
            selection_hash=SHA,
        )
    failure = json.loads((tmp_path / "selection" / "failures" / "gcg.failed.json").read_text(encoding="utf-8"))
    assert failure == {"failure": "failed secondary judgment", "method": "gcg"}
    assert not (tmp_path / "selection" / "methods" / "gcg.complete.json").exists()


def test_serial_runner_stops_after_the_first_method_gate_error(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    calls: list[str] = []

    def fake_gate(**kwargs: object) -> object:
        method = str(kwargs["method"])
        calls.append(method)
        if method == "gcg":
            raise MethodGateError("fixture failure")
        return SimpleNamespace(method=method, primary_judgment_count=153, response_count=51, secondary_judgment_count=153)

    monkeypatch.setattr(cli, "run_method_gate", fake_gate)

    with pytest.raises(MethodGateError, match="fixture failure"):
        cli.run_selected_methods(
            parent_root=tmp_path / "parent",
            selection_root=tmp_path / "selection",
            target_key="qwen2_5_7b",
            samples_per_source=17,
            primary_key="octopus",
            thresholds=THRESHOLDS,
            judge=object(),
            selection_hash=SHA,
        )

    assert calls == ["init", "gcg"]


def test_selected_matrix_analysis_uses_only_completed_method_markers(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    selection = tmp_path / "selection"
    selection.mkdir()
    (selection / "selection_manifest.json").write_text(
        json.dumps({
            "sources": ["harmbench"],
            "target_key": "qwen2_5_7b",
            "samples_per_source": 2,
        }),
        encoding="utf-8",
    )
    for method in ("init", "gcg"):
        (selection / "methods").mkdir(exist_ok=True)
        (selection / "methods" / f"{method}.complete.json").write_text(
            json.dumps({
                "method": method,
                "source_count": 1,
                "response_count": 2,
                "primary_judgment_count": 2,
                "secondary_judgment_count": 2,
            }),
            encoding="utf-8",
        )
        primary = parent / "judgments" / "octopus" / "qwen2_5_7b" / "harmbench" / method / "records.jsonl"
        secondary = selection / "judgments" / "qwen32_compat" / "qwen2_5_7b" / "harmbench" / method / "records.jsonl"
        rows = [
            {
                "schema_version": "reviewer_eval.v1", "run_id": "fixture", "config_hash": SHA,
                "sample_id": f"harmbench:{index}", "source": "harmbench", "method": method,
                "checkpoint": 0 if method == "init" else 100, "target_key": "qwen2_5_7b",
                "judge_key": "octopus", "judge_revision": "fixture", "unsafe_score": float(index),
                "unsafe_label": bool(index), "refusal_label": None, "threshold": 0.5,
                "raw_output": "fixture", "status": "complete", "failure_kind": None, "failure_reason": None,
            }
            for index in range(2)
        ]
        _write_jsonl(primary, rows)
        _write_jsonl(secondary, [{**row, "judge_key": "qwen32_compat"} for row in rows])

    completed = subprocess.run(
        [
            sys.executable, str(ANALYSIS_SCRIPT), "--parent-root", str(parent),
            "--selection-root", str(selection), "--primary-judge", "octopus",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = (selection / "analysis" / "summary.csv").read_text(encoding="utf-8")
    assert summary.count("harmbench") == 4
    assert "init" in summary and "gcg" in summary
    assert "unmarked" not in summary
