from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmark.safety_eval.materialization_ablation import MaterializationPair
from benchmark.safety_eval.materialization_judging import PairedJudgmentRecord


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_materialization_judging.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_materialization_judging_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(sample_id: str = "harmbench:1", *, status: str = "complete") -> MaterializationPair:
    failed = status == "failed"
    return MaterializationPair.model_validate({
        "source": "harmbench",
        "sample_id": sample_id,
        "branch": "High-Value",
        "optimization_checkpoint": 100,
        "state_sha256": None if failed else "a" * 64,
        "model_key": "qwen2_5_7b",
        "model_revision": "local",
        "initial_discrete_prompt": "initial",
        "reference_intent": "intent",
        "continuous_response": "" if failed else "continuous response",
        "materialized_text": "" if failed else "materialized prompt",
        "materialized_response": "" if failed else "materialized response",
        "editable_projected_token_ids": [] if failed else [1],
        "projected_token_ids": [] if failed else [1],
        "retokenized_token_ids": [] if failed else [1],
        "projection_cosines": [] if failed else [0.9],
        "roundtrip_exact_match": not failed,
        "projected_length": 0 if failed else 1,
        "retokenized_length": 0 if failed else 1,
        "max_new_tokens": 512,
        "status": status,
        "error": "generation failed" if failed else None,
        "judgments": {},
    })


def _config(tmp_path: Path) -> Path:
    source_root = tmp_path / "matched"
    output_root = tmp_path / "ablation"
    source_root.mkdir()
    output_root.mkdir()
    (source_root / "locked_config.json").write_text(json.dumps({
        "judging": {
            "primary": {"key": "octopus_seval_14b", "threshold": 0.5},
            "secondary": {
                "key": "qwen32_compat",
                "threshold": 0.5,
                "endpoint": "http://localhost:8001/v1",
                "model": "qwen3-32b-awq",
            },
        },
        "models": {
            "octopus": {
                "key": "octopus_seval_14b",
                "local_path": str(tmp_path / "octopus"),
                "repo_id": "Alibaba-AAIG/Octopus-SEval-14B",
            }
        },
    }), encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""source_root: {source_root}
output_root: {output_root}
model:
  key: qwen2_5_7b
  revision: local
judges:
  primary:
    key: octopus_seval_14b
    repo_id: Alibaba-AAIG/Octopus-SEval-14B
    local_path: {tmp_path / 'octopus'}
    threshold: 0.5
  secondary:
    key: qwen32_compat
    endpoint: http://localhost:8001/v1
    model: qwen3-32b-awq
    threshold: 0.5
""",
        encoding="utf-8",
    )
    return path


def _write_pairs(path: Path) -> None:
    path.write_text(
        _pair().model_dump_json() + "\n" + _pair("failed", status="failed").model_dump_json() + "\n",
        encoding="utf-8",
    )


def test_load_judging_config_matches_locked_judges(tmp_path: Path) -> None:
    module = _load_script()

    config = module.load_judging_config(_config(tmp_path))

    assert config.target_key == "qwen2_5_7b"
    assert config.primary.key == "octopus_seval_14b"
    assert config.secondary.key == "qwen32_compat"
    assert config.primary.threshold == config.secondary.threshold == pytest.approx(0.5)


def test_load_judging_config_rejects_threshold_drift(tmp_path: Path) -> None:
    module = _load_script()
    path = _config(tmp_path)
    path.write_text(path.read_text().replace("threshold: 0.5", "threshold: 0.4", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen matched judge thresholds"):
        module.load_judging_config(path)


def test_dry_run_reports_pending_without_loading_judge(tmp_path: Path, capsys) -> None:
    module = _load_script()
    config = _config(tmp_path)
    pairs = tmp_path / "pairs.jsonl"
    _write_pairs(pairs)

    assert module.main([
        "--config", str(config),
        "--pairs", str(pairs),
        "--judge", "octopus_seval_14b",
        "--dry-run",
    ]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "complete_generation_pairs": 1,
        "existing_complete_judgments": 0,
        "existing_failed_judgments": 0,
        "failed_generation_pairs": 1,
        "judge_key": "octopus_seval_14b",
        "pending_pairs": 1,
    }


def test_run_with_judge_writes_strict_paired_records(tmp_path: Path) -> None:
    module = _load_script()
    config = module.load_judging_config(_config(tmp_path))
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs(pairs_path)

    class Judge:
        key = "octopus_seval_14b"
        revision = "fixture"

        def judge_response(self, **kwargs):
            materialized = "materialized" in kwargs["method"]
            return type("Result", (), {
                "status": "complete",
                "unsafe_score": 0.9 if materialized else 0.1,
                "unsafe_label": materialized,
                "raw_output": "unsafe" if materialized else "safe",
                "failure_reason": None,
                "judge_key": self.key,
                "judge_revision": self.revision,
                "threshold": kwargs["threshold"],
            })()

    output = tmp_path / "judgments.jsonl"
    report = module.run_with_judge(config, pairs_path, output, judge=Judge())

    assert report["written"] == 1
    record = PairedJudgmentRecord.model_validate_json(output.read_text().strip())
    assert record.continuous is False
    assert record.materialized is True


def test_run_with_judge_rejects_existing_complete_provenance_drift(tmp_path: Path) -> None:
    module = _load_script()
    config = module.load_judging_config(_config(tmp_path))
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs(pairs_path)
    output = tmp_path / "judgments.jsonl"
    output.write_text(json.dumps({
        "source": "harmbench",
        "sample_id": "harmbench:1",
        "branch": "High-Value",
        "optimization_checkpoint": 100,
        "run_id": "materialization:old",
        "config_hash": "c" * 64,
        "target_key": "qwen2_5_7b",
        "target_revision": "local",
        "judge_key": "octopus_seval_14b",
        "judge_revision": "old-revision",
        "threshold": 0.5,
        "continuous_score": 0.1,
        "materialized_score": 0.9,
        "continuous": False,
        "materialized": True,
        "continuous_raw_output": "safe",
        "materialized_raw_output": "unsafe",
        "status": "complete",
        "error": None,
    }) + "\n", encoding="utf-8")

    judge = type("Judge", (), {"key": "octopus_seval_14b", "revision": "new-revision"})()
    with pytest.raises(ValueError, match="provenance"):
        module.run_with_judge(config, pairs_path, output, judge=judge)


def test_secondary_runtime_endpoint_override_preserves_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    module = _load_script()
    config_path = _config(tmp_path)
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs(pairs_path)
    captured: dict[str, object] = {}

    class FakeJudge:
        key = "qwen32_compat"
        revision = "endpoint:http://127.0.0.1:8002/v1;model:qwen3-32b-awq;temperature:0"

        def __init__(self, *, endpoint: str, model: str) -> None:
            captured["endpoint"] = endpoint
            captured["model"] = model

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_run(config, pairs, output, *, judge, max_pairs):
        captured["frozen_endpoint"] = config.secondary.endpoint
        captured["judge_revision"] = judge.revision
        return {"written": 1}

    monkeypatch.setattr(module, "Qwen32CompatJudge", FakeJudge)
    monkeypatch.setattr(module, "run_with_judge", fake_run)

    assert module.main([
        "--config", str(config_path),
        "--pairs", str(pairs_path),
        "--judge", "qwen32_compat",
        "--runtime-endpoint", "http://127.0.0.1:8002/v1",
    ]) == 0

    assert captured == {
        "endpoint": "http://127.0.0.1:8002/v1",
        "model": "qwen3-32b-awq",
        "frozen_endpoint": "http://localhost:8001/v1",
        "judge_revision": "endpoint:http://127.0.0.1:8002/v1;model:qwen3-32b-awq;temperature:0",
    }
    assert json.loads(capsys.readouterr().out)["runtime_endpoint"] == "http://127.0.0.1:8002/v1"


def test_primary_runtime_model_path_override_records_resolved_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    module = _load_script()
    config_path = _config(tmp_path)
    pairs_path = tmp_path / "pairs.jsonl"
    _write_pairs(pairs_path)
    runtime_path = tmp_path / "mounted-octopus"
    captured: dict[str, object] = {}

    class Handle:
        model = object()
        tokenizer = object()

        def close(self) -> None:
            captured["closed"] = True

    class FakeJudge:
        key = "octopus_seval_14b"

        def __init__(self, *, model, tokenizer, revision: str) -> None:
            self.revision = revision
            captured["revision"] = revision

    def fake_validate(path):
        captured["runtime_path"] = Path(path)
        return SimpleNamespace(path=runtime_path, revision="local-sha256:resolved")

    def fake_run(config, pairs, output, *, judge, max_pairs):
        captured["frozen_path"] = config.primary.local_path
        return {"written": 1}

    monkeypatch.setattr(module, "validate_model_assets", fake_validate)
    monkeypatch.setattr(module, "load_local_qwen", lambda resolved: Handle())
    monkeypatch.setattr(module, "OctopusLocalJudge", FakeJudge)
    monkeypatch.setattr(module, "run_with_judge", fake_run)

    assert module.main([
        "--config", str(config_path),
        "--pairs", str(pairs_path),
        "--judge", "octopus_seval_14b",
        "--runtime-model-path", str(runtime_path),
    ]) == 0

    assert captured["runtime_path"] == runtime_path
    assert captured["frozen_path"] == tmp_path / "octopus"
    assert captured["revision"] == "local-sha256:resolved"
    assert captured["closed"] is True
    assert json.loads(capsys.readouterr().out)["runtime_model_path"] == str(runtime_path)
