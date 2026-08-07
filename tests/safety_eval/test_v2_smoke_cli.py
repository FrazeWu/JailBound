from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmark.safety_eval.io import canonical_hash
from benchmark.safety_eval.pipeline import StageSummary
from benchmark.safety_eval.schema import RecordStatus, TransportType, V2MaterializationRecord, token_ids_sha256


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_v2_smoke.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("v2_smoke_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(root: Path, *, smoke_mode: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            smoke_mode=smoke_mode,
            output_root=root,
            locked_config_name="locked_config.json",
            schema_version="reviewer_eval.v2",
            seed=20260725,
            attention_implementation="eager",
        ),
        data=SimpleNamespace(sources=["advbench"], samples_per_source=1),
        optimization=SimpleNamespace(
            methods=["random_mutation"],
            checkpoints=[0, 100],
            update_budget=100,
            dual_branch_updates={},
            candidate_cap=3,
            prefix_tokens=2,
            prefix_initialization=SimpleNamespace(token_text="!"),
            learning_rate=0.001,
            gbda_learning_rate=0.2,
            gcg_search_width=2,
            lambda_fol=0.1,
            epsilon=0.1,
            gamma_z=0.01,
            gamma_u=0.01,
            grad_clip=1.0,
            answer_anchors=["answer"],
            refusal_anchors=["refusal"],
        ),
        models=SimpleNamespace(
            surrogate=SimpleNamespace(local_path=root / "surrogate"),
            targets=[SimpleNamespace(key="target", local_path=root / "target")],
            octopus=SimpleNamespace(local_path=root / "octopus"),
        ),
        judging=SimpleNamespace(
            primary=SimpleNamespace(threshold=0.5, threshold_offsets=[0.0]),
            secondary=SimpleNamespace(model="secondary", endpoint="http://judge", local_path=root / "secondary", threshold=0.5, threshold_offsets=[0.0]),
            max_new_tokens=32,
        ),
    )


def test_smoke_runner_requires_explicit_smoke_mode(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "load_v2_config", lambda _: _config(tmp_path, smoke_mode=False))

    with pytest.raises(ValueError, match="smoke_mode"):
        cli.main(["--config", "fixture.yaml"])


def test_smoke_runner_runs_stages_in_paper_order(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli, "load_v2_config", lambda _: config)
    monkeypatch.setattr(cli, "build_manifests", lambda *_args, **_kwargs: calls.append("manifest"))
    monkeypatch.setattr(cli, "_optimize", lambda *_args, **_kwargs: calls.append("optimize"))
    monkeypatch.setattr(cli, "_materialize", lambda *_args, **_kwargs: calls.append("materialize") or [{"schema_version": "reviewer_eval.v2"}])
    monkeypatch.setattr(cli, "_generate", lambda *_args, **_kwargs: calls.append("target") or [{"schema_version": "reviewer_eval.v2"}])
    monkeypatch.setattr(cli, "_judge_primary", lambda *_args, **_kwargs: calls.append("primary"))
    monkeypatch.setattr(cli, "_judge_secondary", lambda *_args, **_kwargs: calls.append("secondary"))
    monkeypatch.setattr(cli, "_write_analysis", lambda *_args, **_kwargs: calls.append("analysis"))

    assert cli.main(["--config", "fixture.yaml"]) == 0
    assert calls == ["manifest", "optimize", "materialize", "target", "primary", "secondary", "analysis"]


def test_smoke_runner_launches_secondary_only_after_primary(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli, "load_v2_config", lambda _: config)
    monkeypatch.setattr(cli, "build_manifests", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_optimize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_materialize", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(cli, "_generate", lambda *_args, **_kwargs: [{}])
    monkeypatch.setattr(cli, "_judge_primary", lambda *_args, **_kwargs: calls.append("primary"))

    @contextmanager
    def server(*_args, **_kwargs):
        calls.append("server-start")
        yield
        calls.append("server-stop")

    monkeypatch.setattr(cli, "_secondary_judge_server", server)
    monkeypatch.setattr(cli, "_judge_secondary", lambda *_args, **_kwargs: calls.append("secondary"))
    monkeypatch.setattr(cli, "_write_analysis", lambda *_args, **_kwargs: calls.append("analysis"))

    assert cli.main(["--config", "fixture.yaml", "--launch-secondary-vllm"]) == 0
    assert calls == ["primary", "server-start", "secondary", "server-stop", "analysis"]


def test_smoke_analysis_writes_empty_paired_table_without_an_init_baseline(
    monkeypatch, tmp_path: Path
) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "write_materialization_summaries", lambda _: tmp_path / "materialization.csv")
    monkeypatch.setattr(cli, "write_judgment_summaries", lambda _: tmp_path / "judgments.csv")
    monkeypatch.setattr(
        cli,
        "load_judgment_rows",
        lambda _: [{"method": "random_mutation"}],
    )
    monkeypatch.setattr(
        cli,
        "write_paired_judgment_differences",
        lambda _: pytest.fail("single-method smoke must not request an init comparison"),
    )

    cli._write_analysis(None, tmp_path)

    assert (tmp_path / "analysis" / "paired_asr.csv").read_text(encoding="utf-8") == (
        "Judge,Target,Source,Method,Threshold,Denominator,Net ASR change,"
        "Net ASR change (%),Method-only,Init-only,McNemar p,Schema,Branch,Transport,"
        "Judge revision,Target revision,Target tokenizer SHA256,Materialization SHA256s,Provenance\n"
    )


def test_smoke_generate_returns_only_current_target_and_materialization_responses(
    monkeypatch, tmp_path: Path
) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    materialization_payload = {
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "a" * 64,
        "sample_id": "advbench:1", "source": "advbench", "method": "random_mutation",
        "branch": "random_mutation", "step": 1, "transport": TransportType.text,
        "state_sha256": "b" * 64, "surrogate_tokenizer_sha256": "c" * 64,
        "surrogate_embedding_sha256": "d" * 64, "editable_positions": (1,),
        "original_token_ids": (3, 4), "projected_z_token_ids": (7,), "projected_u_token_ids": (8,),
        "reconstructed_base_token_ids": (3, 8), "complete_token_ids": (7, 3, 8),
        "frozen_positions_unchanged": True, "span_boundary_expansions": ((0, 1),),
        "full_prompt_similarity": 0.5, "editable_span_similarity": 0.0, "flat_prompt": "fixture",
        "status": RecordStatus.complete, "failure_kind": None, "failure_reason": None,
    }
    materialization = V2MaterializationRecord.model_validate({
        **materialization_payload, "materialization_sha256": canonical_hash(materialization_payload),
    })
    token_hash = token_ids_sha256(materialization.complete_token_ids)
    response_payload = {
        "schema_version": "reviewer_eval.v2", "run_id": materialization.run_id,
        "config_hash": materialization.config_hash, "sample_id": materialization.sample_id,
        "source": materialization.source, "method": materialization.method,
        "checkpoint": materialization.step, "target_key": "target", "target_revision": "current",
        "prompt_hash": token_hash, "response": "fixture", "input_tokens": 3, "generated_tokens": 1,
        "status": RecordStatus.complete, "failure_kind": None, "failure_reason": None,
        "branch": materialization.branch, "state_step": materialization.step,
        "transport": materialization.transport, "materialization_sha256": materialization.materialization_sha256,
        "target_tokenizer_sha256": "c" * 64, "executed_token_ids_sha256": token_hash,
    }
    response_path = tmp_path / "responses" / "target" / "advbench" / "random_mutation" / "records.jsonl"
    response_path.parent.mkdir(parents=True)
    stale = {**response_payload, "target_revision": "stale"}
    response_path.write_text(
        json.dumps(stale) + "\n" + json.dumps(response_payload) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        cli, "generate_v2_materialized_records_from_local_assets",
        lambda *_args, **_kwargs: StageSummary(1, 0, 0),
    )
    monkeypatch.setattr(
        cli, "validate_model_assets",
        lambda *_args: SimpleNamespace(revision="current", tokenizer_hash="c" * 64),
    )

    responses = cli._generate(config, tmp_path, [materialization.model_dump(mode="json")])

    assert len(responses) == 1
    assert responses[0]["target_revision"] == "current"

    response_path.write_text(
        json.dumps(response_payload) + "\n" + json.dumps(response_payload) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="duplicates a current materialization"):
        cli._generate(config, tmp_path, [materialization.model_dump(mode="json")])


def test_smoke_secondary_judge_uses_the_local_snapshot_revision(
    monkeypatch, tmp_path: Path
) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    captured: dict[str, object] = {}

    class Judge:
        key = "qwen32_compat"
        revision = "local-sha256:" + "a" * 64

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def __enter__(self) -> "Judge":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        cli,
        "validate_model_assets",
        lambda path: SimpleNamespace(revision="local-sha256:" + "a" * 64),
    )
    monkeypatch.setattr(cli, "Qwen32CompatJudge", Judge)
    monkeypatch.setattr(cli, "judge_response_records", lambda *_args, **_kwargs: StageSummary(1, 0, 0))

    cli._judge_secondary(config, tmp_path, [{"schema_version": "reviewer_eval.v2"}])

    assert captured["revision"] == "local-sha256:" + "a" * 64
    assert captured["endpoint"] == config.judging.secondary.endpoint


def test_smoke_preflight_rejects_legacy_v2_response_without_execution_provenance(
    tmp_path: Path,
) -> None:
    cli = _load_cli()
    response_path = tmp_path / "responses" / "target" / "source" / "method" / "records.jsonl"
    response_path.parent.mkdir(parents=True)
    response_path.write_text('{"schema_version":"reviewer_eval.v2"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="legacy v2 provenance"):
        cli._reject_legacy_v2_provenance(tmp_path)
