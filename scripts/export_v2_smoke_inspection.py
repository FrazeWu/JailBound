#!/usr/bin/env python3
"""Collect every reviewer-v2 smoke artifact into one inspectable JSON file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _to_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _tensor_metadata(value: Any, *, prefix: str = "") -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    if isinstance(value, torch.Tensor):
        metadata[prefix] = {"dtype": str(value.dtype), "shape": list(value.shape)}
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            metadata.update(_tensor_metadata(item, prefix=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            metadata.update(_tensor_metadata(item, prefix=child))
    return metadata


def export(output_root: Path, destination: Path) -> dict[str, object]:
    root = output_root.resolve()
    if not root.is_dir():
        raise ValueError(f"output root does not exist: {root}")
    destination = destination.resolve()
    json_artifacts: dict[str, object] = {}
    jsonl_artifacts: dict[str, object] = {}
    text_artifacts: dict[str, str] = {}
    checkpoint_states: dict[str, object] = {}
    other_files: dict[str, object] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == destination:
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".json":
            json_artifacts[relative] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".jsonl":
            jsonl_artifacts[relative] = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint_states[relative] = {
                "sha256": _sha256(path),
                "tensor_metadata": _tensor_metadata(payload),
                "payload": _to_json(payload),
            }
        elif path.suffix == ".log":
            text_artifacts[relative] = path.read_text(encoding="utf-8", errors="replace")
        else:
            other_files[relative] = {"sha256": _sha256(path), "bytes": path.stat().st_size}

    return {
        "output_root": str(root),
        "json_artifacts": json_artifacts,
        "jsonl_artifacts": jsonl_artifacts,
        "checkpoint_states": checkpoint_states,
        "text_artifacts": text_artifacts,
        "other_files": other_files,
    }


def main() -> int:
    args = parse_args()
    payload = export(args.output_root, args.destination)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
