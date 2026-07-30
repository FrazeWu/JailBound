from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "inventory_safety_eval_legacy.py"


def _module():
    spec = importlib.util.spec_from_file_location("legacy_inventory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_inventory_detects_changed_artifact(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    artifact = root / "manifest.json"
    artifact.write_text("original", encoding="utf-8")

    inventory = module.build_inventory((root,))
    module.verify_inventory(inventory)

    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="legacy artifact changed"):
        module.verify_inventory(inventory)


def test_inventory_orders_roots_and_relative_files_and_records_sha256(tmp_path: Path) -> None:
    module = _module()
    first = tmp_path / "z-root"
    second = tmp_path / "a-root"
    (first / "nested").mkdir(parents=True)
    second.mkdir()
    (first / "nested" / "z.txt").write_text("zeta", encoding="utf-8")
    (first / "a.txt").write_text("alpha", encoding="utf-8")
    (second / "b.txt").write_text("bravo", encoding="utf-8")

    inventory = module.build_inventory((first, second))

    assert [entry["root"] for entry in inventory["roots"]] == [str(second.resolve()), str(first.resolve())]
    first_entry = inventory["roots"][1]
    assert [entry["path"] for entry in first_entry["files"]] == ["a.txt", "nested/z.txt"]
    assert first_entry["files"][0]["sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert first_entry["files"][0]["size"] == len(b"alpha")
    assert isinstance(first_entry["files"][0]["mtime_ns"], int)


def test_verify_inventory_detects_missing_and_new_files(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    artifact = root / "manifest.json"
    artifact.write_text("original", encoding="utf-8")
    inventory = module.build_inventory((root,))

    artifact.unlink()
    with pytest.raises(ValueError, match="legacy artifact missing"):
        module.verify_inventory(inventory)

    artifact.write_text("original", encoding="utf-8")
    (root / "new.json").write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="new legacy artifact"):
        module.verify_inventory(inventory)


def test_build_inventory_rejects_file_changed_while_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    artifact = root / "manifest.json"
    artifact.write_text("original", encoding="utf-8")
    original_hash_descriptor = module._hash_descriptor

    def mutate_after_hash(descriptor: int) -> str:
        digest = original_hash_descriptor(descriptor)
        artifact.write_text("changed after hashing", encoding="utf-8")
        return digest

    monkeypatch.setattr(module, "_hash_descriptor", mutate_after_hash)

    with pytest.raises(ValueError, match="changed while hashing"):
        module.build_inventory((root,))


def test_build_inventory_rejects_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    artifact = root / "manifest.json"
    artifact.write_text("original", encoding="utf-8")
    original_stat = artifact.stat()
    original_hash_descriptor = module._hash_descriptor

    def mutate_without_size_or_mtime_change(descriptor: int) -> str:
        digest = original_hash_descriptor(descriptor)
        artifact.write_text("modified", encoding="utf-8")
        os.utime(artifact, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return digest

    monkeypatch.setattr(module, "_hash_descriptor", mutate_without_size_or_mtime_change)

    with pytest.raises(ValueError, match="changed while hashing"):
        module.build_inventory((root,))


def test_build_inventory_rejects_atomic_file_replacement_while_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    artifact = root / "manifest.json"
    artifact.write_text("original", encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text("modified", encoding="utf-8")
    original_hash_descriptor = module._hash_descriptor
    replaced = False

    def replace_after_hash(descriptor: int) -> str:
        nonlocal replaced
        digest = original_hash_descriptor(descriptor)
        if not replaced:
            os.replace(replacement, artifact)
            replaced = True
        return digest

    monkeypatch.setattr(module, "_hash_descriptor", replace_after_hash)

    with pytest.raises(ValueError, match="changed while hashing"):
        module.build_inventory((root,))


def test_build_inventory_rejects_symlink_substitution_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    artifact = root / "manifest.json"
    artifact.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    original_iter_relative_files = module._iter_relative_files

    def substitute_after_check(root_descriptor: int, prefix: Path = Path()):
        for relative_path in original_iter_relative_files(root_descriptor, prefix):
            path.unlink()
            path.symlink_to(outside)
            yield relative_path

    path = artifact
    monkeypatch.setattr(module, "_iter_relative_files", substitute_after_check)

    with pytest.raises(ValueError, match="symlink|changed while hashing"):
        module.build_inventory((root,))


@pytest.mark.parametrize("unsafe_root", (Path("/"), Path.home(), ROOT))
def test_build_inventory_rejects_unsafe_roots(unsafe_root: Path) -> None:
    module = _module()

    with pytest.raises(ValueError, match="unsafe inventory root"):
        module.build_inventory((unsafe_root,))


def test_build_inventory_rejects_broad_system_root() -> None:
    module = _module()

    with pytest.raises(ValueError, match="unsafe inventory root"):
        module.build_inventory((Path("/tmp"),))


@pytest.mark.parametrize(
    "unsafe_root",
    (Path("/usr/local/share"), Path("/etc/systemd/system"), Path("/var/lib/apt/lists")),
)
def test_build_inventory_rejects_nested_system_trees(unsafe_root: Path) -> None:
    module = _module()
    if not unsafe_root.is_dir():
        pytest.skip(f"system fixture does not exist: {unsafe_root}")

    with pytest.raises(ValueError, match="unsafe inventory root"):
        module.build_inventory((unsafe_root,))


def test_build_inventory_rejects_overlapping_roots(tmp_path: Path) -> None:
    module = _module()
    parent = tmp_path / "legacy"
    child = parent / "nested"
    child.mkdir(parents=True)

    with pytest.raises(ValueError, match="overlap"):
        module.build_inventory((parent, child))


def test_verify_inventory_rejects_non_string_relative_path(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    inventory = {
        "version": module.INVENTORY_VERSION,
        "roots": [{"root": str(root.resolve()), "files": [{"path": []}]}],
    }

    with pytest.raises(ValueError, match="invalid legacy inventory file entry"):
        module.verify_inventory(inventory)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda inventory: inventory.update({"extra": True}),
        lambda inventory: inventory["roots"][0].update({"extra": True}),
        lambda inventory: inventory.update({"version": True}),
    ),
    ids=("top-level-extra", "root-extra", "boolean-version"),
)
def test_verify_inventory_requires_exact_document_schema(tmp_path: Path, mutate) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    inventory = module.build_inventory((root,))
    mutate(inventory)

    with pytest.raises(ValueError, match="invalid legacy inventory|unsupported legacy inventory version"):
        module.verify_inventory(inventory)


def test_verify_inventory_rejects_overlapping_root_set(tmp_path: Path) -> None:
    module = _module()
    parent = tmp_path / "legacy"
    child = parent / "nested"
    child.mkdir(parents=True)
    inventory = {
        "version": module.INVENTORY_VERSION,
        "roots": [
            {"root": str(parent.resolve()), "files": []},
            {"root": str(child.resolve()), "files": []},
        ],
    }

    with pytest.raises(ValueError, match="overlap"):
        module.verify_inventory(inventory)


def test_validate_destination_rejects_lexical_path_inside_root_when_symlink_points_outside(tmp_path: Path) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    outside = tmp_path / "outside.json"
    destination = root / "inventory.json"
    destination.symlink_to(outside)

    with pytest.raises(ValueError, match="inside an inventory root"):
        module._validate_destination(destination, (root,))


def test_build_inventory_rejects_root_replacement_during_descriptor_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "manifest.json").write_text("original", encoding="utf-8")
    displaced = tmp_path / "displaced"
    original_iter_relative_files = module._iter_relative_files
    replaced = False

    def replace_root_after_scan(root_descriptor: int):
        nonlocal replaced
        relative_paths = tuple(original_iter_relative_files(root_descriptor))
        if not replaced:
            root.rename(displaced)
            root.mkdir()
            (root / "manifest.json").write_text("original", encoding="utf-8")
            (root / "unrecorded.json").write_text("new", encoding="utf-8")
            replaced = True
        yield from relative_paths

    monkeypatch.setattr(module, "_iter_relative_files", replace_root_after_scan)

    with pytest.raises(ValueError, match="changed while inventorying"):
        module.build_inventory((root,))


def test_cli_write_and_verify_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "manifest.json").write_text("original", encoding="utf-8")
    destination = tmp_path / "inventory.json"

    write = subprocess.run(
        [sys.executable, str(SCRIPT), "write", "--root", str(root), "--destination", str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert write.returncode == 0, write.stderr
    assert json.loads(destination.read_text(encoding="utf-8"))["roots"][0]["root"] == str(root.resolve())

    verify = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--inventory", str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr


def test_cli_rejects_destination_inside_inventory_root(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "manifest.json").write_text("original", encoding="utf-8")
    destination = root / "inventory.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "write", "--root", str(root), "--destination", str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "inside an inventory root" in result.stderr
    assert not destination.exists()
