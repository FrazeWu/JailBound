from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


INVENTORY_VERSION = 1


def _resolve_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"inventory root is not a directory: {resolved}")
    if resolved in (Path("/"), Path.home().resolve()) or (resolved / ".git").exists():
        raise ValueError(f"unsafe inventory root: {resolved}")
    return resolved


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"symlinked legacy artifact is not supported: {path}")
        if path.is_file():
            yield path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(root: Path, path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = _hash_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"legacy artifact changed while hashing: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": digest,
    }


def build_inventory(roots: Iterable[Path]) -> dict[str, Any]:
    resolved_roots = sorted({_resolve_root(Path(root)) for root in roots}, key=str)
    if not resolved_roots:
        raise ValueError("at least one inventory root is required")
    return {
        "version": INVENTORY_VERSION,
        "roots": [
            {
                "root": str(root),
                "files": [_file_entry(root, path) for path in _iter_files(root)],
            }
            for root in resolved_roots
        ],
    }


def verify_inventory(inventory: dict[str, Any]) -> None:
    if inventory.get("version") != INVENTORY_VERSION:
        raise ValueError("unsupported legacy inventory version")
    roots = inventory.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ValueError("legacy inventory has no roots")

    for root_entry in roots:
        if not isinstance(root_entry, dict) or not isinstance(root_entry.get("root"), str):
            raise ValueError("invalid legacy inventory root")
        root = _resolve_root(Path(root_entry["root"]))
        expected_files = root_entry.get("files")
        if not isinstance(expected_files, list):
            raise ValueError("invalid legacy inventory files")
        expected_by_path = {entry.get("path"): entry for entry in expected_files if isinstance(entry, dict)}
        if len(expected_by_path) != len(expected_files) or not all(isinstance(path, str) for path in expected_by_path):
            raise ValueError("invalid legacy inventory file entry")

        current_paths = {path.relative_to(root).as_posix(): path for path in _iter_files(root)}
        for relative_path in sorted(set(expected_by_path) - set(current_paths)):
            raise ValueError(f"legacy artifact missing: {root / relative_path}")
        for relative_path in sorted(set(current_paths) - set(expected_by_path)):
            raise ValueError(f"new legacy artifact: {current_paths[relative_path]}")

        for relative_path in sorted(expected_by_path):
            expected = expected_by_path[relative_path]
            current = _file_entry(root, current_paths[relative_path])
            if current != expected:
                raise ValueError(f"legacy artifact changed: {current_paths[relative_path]}")


def write_inventory(destination: Path, inventory: dict[str, Any]) -> None:
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_inventory(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("legacy inventory must be a JSON object")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory immutable legacy reviewer artifacts.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    write = subcommands.add_parser("write", help="write an inventory")
    write.add_argument("--root", action="append", required=True, type=Path)
    write.add_argument("--destination", required=True, type=Path)
    verify = subcommands.add_parser("verify", help="verify an existing inventory")
    verify.add_argument("--inventory", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write":
            write_inventory(args.destination, build_inventory(args.root))
        else:
            verify_inventory(_read_inventory(args.inventory))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
