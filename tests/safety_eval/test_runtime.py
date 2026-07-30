from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.runtime import (
    PreflightError,
    git_provenance,
    lock_runtime_config,
    validate_model_assets,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_model_snapshot(path: Path, names: tuple[str, ...]) -> None:
    path.mkdir()
    for index, name in enumerate(names):
        asset = path / name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text(f"asset-{index}:{name}\n", encoding="utf-8")


def test_validate_model_assets_accepts_complete_local_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    resolved = validate_model_assets(model)
    assert resolved.revision.startswith("local-sha256:")
    assert resolved.tokenizer_hash


@pytest.mark.parametrize(
    "changed_asset",
    (
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
        "nested/loader_metadata.json",
    ),
)
def test_validate_model_assets_revision_covers_every_regular_snapshot_file(
    tmp_path: Path, changed_asset: str
) -> None:
    model = tmp_path / "model"
    names = (
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
        "nested/loader_metadata.json",
    )
    _write_model_snapshot(model, names)
    original = validate_model_assets(model)

    with (model / changed_asset).open("a", encoding="utf-8") as stream:
        stream.write("changed\n")

    changed = validate_model_assets(model)
    assert changed.revision != original.revision


@pytest.mark.parametrize(
    "changed_asset",
    (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ),
)
def test_validate_model_assets_tokenizer_hash_covers_tokenizer_loader_inputs(
    tmp_path: Path, changed_asset: str
) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(
        model,
        (
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "model.safetensors",
        ),
    )
    original = validate_model_assets(model)

    with (model / changed_asset).open("a", encoding="utf-8") as stream:
        stream.write("changed\n")

    assert validate_model_assets(model).tokenizer_hash != original.tokenizer_hash


def test_validate_model_assets_identity_is_independent_of_root_and_creation_order(
    tmp_path: Path,
) -> None:
    names = (
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "model.safetensors",
        "nested/metadata.json",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_model_snapshot(first, names)
    _write_model_snapshot(second, tuple(reversed(names)))
    for name in names:
        (second / name).write_bytes((first / name).read_bytes())

    first_resolved = validate_model_assets(first)
    second_resolved = validate_model_assets(second)

    assert first_resolved.revision == second_resolved.revision
    assert first_resolved.tokenizer_hash == second_resolved.tokenizer_hash


def test_validate_model_assets_identity_ignores_downloader_metadata(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(
        model, ("config.json", "tokenizer.json", "model.safetensors")
    )
    metadata = model / ".cache/huggingface/download/config.json.lock"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("download-state-a\n", encoding="utf-8")
    original = validate_model_assets(model)

    metadata.write_text("download-state-b\n", encoding="utf-8")

    assert validate_model_assets(model) == original


def test_validate_model_assets_hashes_symlinked_loader_inputs(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    _write_model_snapshot(
        blobs, ("config.json", "tokenizer.json", "model.safetensors")
    )
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).symlink_to(blobs / name)

    original = validate_model_assets(model)
    with (blobs / "model.safetensors").open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    changed = validate_model_assets(model)

    assert changed.revision != original.revision
    assert changed.tokenizer_hash == original.tokenizer_hash


def test_validate_model_assets_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PreflightError, match="incomplete model snapshot"):
        validate_model_assets(model)


def test_validate_model_assets_accepts_vocab_as_tokenizer_asset(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(model, ("config.json", "vocab.json", "model.safetensors"))

    assert validate_model_assets(model).tokenizer_hash


@pytest.mark.parametrize("missing_asset", ("config.json", "vocab.json", "model.safetensors"))
def test_validate_model_assets_requires_each_snapshot_asset_class(
    tmp_path: Path, missing_asset: str
) -> None:
    model = tmp_path / "model"
    names = {"config.json", "vocab.json", "model.safetensors"} - {missing_asset}
    _write_model_snapshot(model, tuple(sorted(names)))

    with pytest.raises(PreflightError, match="incomplete model snapshot"):
        validate_model_assets(model)


def test_validate_model_assets_tokenizer_hash_ignores_weight_changes(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(
        model, ("config.json", "tokenizer.json", "model.safetensors")
    )
    original = validate_model_assets(model)

    with (model / "model.safetensors").open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    changed = validate_model_assets(model)

    assert changed.revision != original.revision
    assert changed.tokenizer_hash == original.tokenizer_hash


def test_lock_runtime_config_writes_content_addressed_identity(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/benchmark/safety_eval_additions.yaml")
    locked = lock_runtime_config(config, output_root=tmp_path, source_hashes={"advbench": "a" * 64})
    assert locked.run_id.startswith("run:")
    assert (tmp_path / "locked_config.json").exists()
    assert json.loads((tmp_path / "run_manifest.json").read_text())["run_id"] == locked.run_id


def test_git_provenance_is_independent_of_process_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(ROOT)
    inside = git_provenance(ROOT)
    monkeypatch.chdir(elsewhere)
    outside = git_provenance(ROOT)

    assert outside == inside
    assert set(outside) == {"git_revision", "git_status_hash"}


def test_lock_runtime_config_is_byte_identical_across_process_cwds(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config(ROOT / "configs/benchmark/safety_eval_additions.yaml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    monkeypatch.chdir(ROOT)
    lock_runtime_config(
        config,
        output_root=first_root,
        source_hashes={"advbench": "a" * 64},
        repository_root=ROOT,
    )
    monkeypatch.chdir(elsewhere)
    lock_runtime_config(
        config,
        output_root=second_root,
        source_hashes={"advbench": "a" * 64},
        repository_root=ROOT,
    )

    for name in ("locked_config.json", "run_manifest.json"):
        assert (first_root / name).read_bytes() == (second_root / name).read_bytes()
