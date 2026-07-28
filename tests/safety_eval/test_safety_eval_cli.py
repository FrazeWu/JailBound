from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_safety_eval.py"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("safety_eval_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(schema_version="reviewer_eval.v1", seed=20260725, output_root=output_root, locked_config_name="locked_config.json"),
        data=SimpleNamespace(sources=["source_a", "source_b"], samples_per_source=3),
        optimization=SimpleNamespace(
            methods=["init", "random_mutation", "zol", "pez", "gbda", "gcg", "jailbound_o_minus", "jailbound_o_plus", "dual_branch"],
            checkpoints=[0, 25, 50, 100],
            update_budget=100,
            dual_branch_updates={"o_minus": 50, "o_plus": 50},
            candidate_cap=3200,
            prefix_tokens=20,
            editable_seed_tokens=20,
            learning_rate=0.001,
            lambda_fol=0.1,
            epsilon=0.1,
            gamma_z=0.01,
            gamma_u=0.01,
            grad_clip=1.0,
            answer_anchors=["answer"],
            refusal_anchors=["refusal"],
        ),
        models=SimpleNamespace(
            surrogate=SimpleNamespace(local_path=output_root / "local-qwen"),
            targets=[SimpleNamespace(key="target_a"), SimpleNamespace(key="target_b")],
        ),
        judging=SimpleNamespace(primary=SimpleNamespace(key="octopus"), secondary=SimpleNamespace(key="qwen32_compat")),
    )


def test_validate_reports_only_safe_config_counts_and_identities(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path / "result")
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    assert cli.main(["validate", "--config", "fixture.yaml"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "judge_keys": ["octopus", "qwen32_compat"],
        "method_ids": ["init", "random_mutation", "zol", "pez", "gbda", "gcg", "jailbound_o_minus", "jailbound_o_plus", "dual_branch"],
        "planned_sample_count": 6,
        "samples_per_source": 3,
        "schema_version": "reviewer_eval.v1",
        "source_count": 2,
        "target_keys": ["target_a", "target_b"],
    }


def test_manifests_delegates_to_existing_builder_without_running_it(monkeypatch, tmp_path: Path) -> None:
    cli = _load_cli()
    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(cli, "_run_manifest_builder", lambda path, candidate_pool: calls.append((path, candidate_pool)) or 0)

    assert cli.main(["manifests", "--config", str(tmp_path / "fixture.yaml"), "--candidate-pool", "17"]) == 0
    assert calls == [(tmp_path / "fixture.yaml", 17)]


def test_status_reports_lock_and_serial_marker_state(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    (tmp_path / "locked_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    marker = tmp_path / "responses" / "target_a"
    marker.mkdir(parents=True)
    (marker / "TARGET_COMPLETE.json").write_text("{}", encoding="utf-8")

    assert cli.main(["status", "--config", "fixture.yaml"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "locked_root": True,
        "optimization_execution": "available",
        "serial_targets": [
            {"complete": True, "target": "target_a"},
            {"complete": False, "target": "target_b"},
        ],
    }


def test_status_returns_failure_when_locked_root_is_missing(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "load_config", lambda path: _config(tmp_path))

    assert cli.main(["status", "--config", "fixture.yaml"]) == 1
    assert json.loads(capsys.readouterr().out)["locked_root"] is False


def test_materialize_delegates_only_after_loading_the_frozen_threshold(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    config.models.semantic_encoder = SimpleNamespace(local_path=tmp_path / "semantic")
    config.semantic = SimpleNamespace(threshold_artifact=tmp_path / "semantic_calibration.json")
    (tmp_path / "semantic_calibration.json").write_text('{"threshold": 0.9}', encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_run_materialization", lambda config, output_root, final_only: {"selected_records": 3, "written_records": 3, "failed_records": 0, "final_only": final_only})

    assert cli.main(["materialize", "--config", "fixture.yaml", "--final-only"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "failed_records": 0,
        "final_only": True,
        "selected_records": 3,
        "written_records": 3,
    }


def test_run_target_delegates_one_configured_target_without_loading_it_in_the_cli(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[tuple[object, Path, str]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "_run_target",
        lambda received_config, output_root, target_key: calls.append((received_config, output_root, target_key))
        or {"target": target_key, "response_count": 3, "primary_judgment_count": 3, "secondary_judgment_count": 3},
        raising=False,
    )

    assert cli.main(["run-target", "--config", "fixture.yaml", "--target", "target_a"]) == 0

    assert calls == [(config, tmp_path, "target_a")]
    assert json.loads(capsys.readouterr().out) == {
        "primary_judgment_count": 3,
        "response_count": 3,
        "secondary_judgment_count": 3,
        "target": "target_a",
    }


def test_run_target_can_be_limited_to_one_configured_source(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[tuple[object, Path, str, str | None]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "_run_target",
        lambda received_config, output_root, target_key, source=None: calls.append(
            (received_config, output_root, target_key, source)
        )
        or {"target": target_key, "source": source, "response_count": 3, "primary_judgment_count": 3, "secondary_judgment_count": 3},
        raising=False,
    )

    assert cli.main(
        ["run-target", "--config", "fixture.yaml", "--target", "target_a", "--source", "source_a"]
    ) == 0

    assert calls == [(config, tmp_path, "target_a", "source_a")]
    assert json.loads(capsys.readouterr().out)["source"] == "source_a"


def test_run_target_can_be_limited_to_explicit_configured_methods(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[tuple[object, Path, str, str | None, tuple[str, ...] | None]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "_run_target",
        lambda received_config, output_root, target_key, source=None, methods=None: calls.append(
            (received_config, output_root, target_key, source, methods)
        )
        or {"target": target_key, "methods": list(methods or ()), "response_count": 3, "primary_judgment_count": 3, "secondary_judgment_count": 3},
        raising=False,
    )

    assert cli.main(
        [
            "run-target", "--config", "fixture.yaml", "--target", "target_a", "--source", "source_a",
            "--method", "init", "--method", "gcg",
        ]
    ) == 0

    assert calls == [(config, tmp_path, "target_a", "source_a", ("init", "gcg"))]
    assert json.loads(capsys.readouterr().out)["methods"] == ["init", "gcg"]


def test_analyze_writes_aggregate_artifacts_without_model_loading(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "write_judgment_summaries", lambda root: root / "analysis" / "judge_sensitivity.csv")
    monkeypatch.setattr(cli, "write_paired_judgment_differences", lambda root: root / "analysis" / "paired_asr.csv")
    monkeypatch.setattr(cli, "write_materialization_summaries", lambda root: root / "analysis" / "materialization_fidelity.csv")

    assert cli.main(["analyze", "--config", "fixture.yaml"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "judge_sensitivity": str(tmp_path / "analysis" / "judge_sensitivity.csv"),
        "materialization_fidelity": str(tmp_path / "analysis" / "materialization_fidelity.csv"),
        "paired_asr": str(tmp_path / "analysis" / "paired_asr.csv"),
    }


def test_run_smoke_delegates_a_content_free_dry_run(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "run_execution",
        lambda request, *, mode: calls.append((request, mode))
        or SimpleNamespace(selected_records=1, completed_records=0, failed_records=0),
    )

    assert cli.main(
        [
            "run-smoke",
            "--config",
            "fixture.yaml",
            "--source",
            "source_a",
            "--method",
            "init",
            "--limit",
            "1",
            "--dry-run",
        ]
    ) == 0

    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out) == {
        "completed_records": 0,
        "failed_records": 0,
        "method": "init",
        "mode": "dry-run",
        "selected_records": 1,
        "source": "source_a",
    }


def test_run_smoke_execute_injects_local_qwen_init_path_at_checkpoint_zero(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[tuple[object, object, object, object]] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    def fake_run(request, *, mode, model_loader=None, executor=None):
        calls.append((request, mode, model_loader, executor))
        return SimpleNamespace(selected_records=1, completed_records=1, failed_records=0)

    monkeypatch.setattr(cli, "run_execution", fake_run)
    injected = object()
    monkeypatch.setattr(cli, "build_local_qwen_tensor_executor", lambda settings: injected)

    assert cli.main(
        [
            "run-smoke",
            "--config",
            "fixture.yaml",
            "--source",
            "source_a",
            "--method",
            "init",
            "--limit",
            "1",
            "--execute",
        ]
    ) == 0

    assert len(calls) == 1
    request, mode, model_loader, executor = calls[0]
    assert request.checkpoints == (0,)
    assert mode is cli.ExecutionMode.smoke
    assert model_loader is cli.load_local_qwen
    assert executor is injected
    assert json.loads(capsys.readouterr().out)["completed_records"] == 1


def test_run_smoke_execute_permits_all_tensor_optimization_methods(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "run_execution",
        lambda request, **kwargs: calls.append(request)
        or SimpleNamespace(selected_records=1, completed_records=4, failed_records=0),
    )

    for method in ("random_mutation", "pez", "gbda", "gcg"):
        assert cli.main(
            ["run-smoke", "--config", "fixture.yaml", "--source", "source_a", "--method", method, "--execute"]
        ) == 0

    assert [request.method for request in calls] == ["random_mutation", "pez", "gbda", "gcg"]
    assert capsys.readouterr().err == ""


def test_run_smoke_execute_permits_tensor_methods_with_configured_checkpoints(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[tuple[object, object, object, object]] = []
    injected = object()
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "build_local_qwen_tensor_executor", lambda settings: injected)

    def fake_run(request, *, mode, model_loader=None, executor=None):
        calls.append((request, mode, model_loader, executor))
        return SimpleNamespace(selected_records=1, completed_records=4, failed_records=0)

    monkeypatch.setattr(cli, "run_execution", fake_run)

    assert cli.main(
        ["run-smoke", "--config", "fixture.yaml", "--source", "source_a", "--method", "dual_branch", "--execute"]
    ) == 0

    request, mode, model_loader, executor = calls[0]
    assert request.checkpoints == (0, 25, 50, 100)
    assert mode is cli.ExecutionMode.smoke
    assert model_loader is cli.load_local_qwen
    assert executor is injected
    assert json.loads(capsys.readouterr().out)["completed_records"] == 4


def test_run_smoke_execute_dry_run_does_not_inject_model_execution(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "load_config", lambda path: _config(tmp_path))
    calls: list[tuple[object, object, object, object]] = []

    def fake_run(request, *, mode, model_loader=None, executor=None):
        calls.append((request, mode, model_loader, executor))
        return SimpleNamespace(selected_records=1, completed_records=0, failed_records=0)

    monkeypatch.setattr(cli, "run_execution", fake_run)

    assert cli.main(
        [
            "run-smoke",
            "--config",
            "fixture.yaml",
            "--source",
            "source_a",
            "--method",
            "init",
            "--execute",
            "--dry-run",
        ]
    ) == 0

    request, mode, model_loader, executor = calls[0]
    assert request.checkpoints == (0, 25, 50, 100)
    assert mode is cli.ExecutionMode.dry_run
    assert model_loader is executor is None
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"


def test_optimize_uses_the_complete_configured_source_budget(monkeypatch, capsys, tmp_path: Path) -> None:
    cli = _load_cli()
    config = _config(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "run_execution",
        lambda request, **kwargs: calls.append(request)
        or SimpleNamespace(selected_records=3, completed_records=12, failed_records=0),
    )

    assert cli.main(["optimize", "--config", "fixture.yaml", "--source", "source_a", "--method", "zol"]) == 0

    assert len(calls) == 1
    assert calls[0].requested_limit == 3
    assert json.loads(capsys.readouterr().out)["completed_records"] == 12
