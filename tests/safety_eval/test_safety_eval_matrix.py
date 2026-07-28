from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_safety_eval_matrix.py"


def _load_matrix() -> ModuleType:
    spec = importlib.util.spec_from_file_location("safety_eval_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_runs_only_the_explicit_source_and_method_filters(monkeypatch, tmp_path: Path) -> None:
    matrix = _load_matrix()
    config = SimpleNamespace(
        data=SimpleNamespace(sources=["source_a", "source_b"], samples_per_source=50),
        optimization=SimpleNamespace(methods=["method_a", "method_b"]),
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(matrix, "load_config", lambda _: config)
    monkeypatch.setattr(matrix, "_records_complete", lambda *_: False)
    monkeypatch.setattr(
        matrix.subprocess,
        "run",
        lambda command, **_: calls.append(command) or SimpleNamespace(returncode=0),
    )

    assert matrix.main(
        [
            "--config",
            "fixture.yaml",
            "--output-root",
            str(tmp_path / "output"),
            "--gpu",
            "1",
            "--source",
            "source_b",
            "--method",
            "method_b",
        ]
    ) == 0

    assert len(calls) == 1
    assert calls[0][-4:] == ["--source", "source_b", "--method", "method_b"]
