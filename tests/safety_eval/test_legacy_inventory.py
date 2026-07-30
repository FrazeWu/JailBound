from __future__ import annotations

import hashlib
import importlib.util
import json
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
    original_hash_file = module._hash_file

    def mutate_after_hash(path: Path) -> str:
        digest = original_hash_file(path)
        path.write_text("changed after hashing", encoding="utf-8")
        return digest

    monkeypatch.setattr(module, "_hash_file", mutate_after_hash)

    with pytest.raises(ValueError, match="changed while hashing"):
        module.build_inventory((root,))


@pytest.mark.parametrize("unsafe_root", (Path("/"), Path.home(), ROOT))
def test_build_inventory_rejects_unsafe_roots(unsafe_root: Path) -> None:
    module = _module()

    with pytest.raises(ValueError, match="unsafe inventory root"):
        module.build_inventory((unsafe_root,))


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
