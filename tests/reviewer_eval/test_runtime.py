from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.reviewer_eval.config import load_config
from benchmark.reviewer_eval.runtime import PreflightError, lock_runtime_config, validate_model_assets


ROOT = Path(__file__).resolve().parents[2]


def test_validate_model_assets_accepts_complete_local_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    resolved = validate_model_assets(model)
    assert resolved.revision.startswith("local-sha256:")
    assert resolved.tokenizer_hash


def test_validate_model_assets_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PreflightError, match="incomplete model snapshot"):
        validate_model_assets(model)


def test_lock_runtime_config_writes_content_addressed_identity(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/benchmark/reviewer_additions.yaml")
    locked = lock_runtime_config(config, output_root=tmp_path, source_hashes={"advbench": "a" * 64})
    assert locked.run_id.startswith("run:")
    assert (tmp_path / "locked_config.json").exists()
    assert json.loads((tmp_path / "run_manifest.json").read_text())["run_id"] == locked.run_id
