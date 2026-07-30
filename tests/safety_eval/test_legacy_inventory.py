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
    original_iter_files = module._iter_files

    def substitute_after_check(inventory_root: Path):
        for path in original_iter_files(inventory_root):
            path.unlink()
            path.symlink_to(outside)
            yield path

    monkeypatch.setattr(module, "_iter_files", substitute_after_check)

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
