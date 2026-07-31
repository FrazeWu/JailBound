"""Offline-only runtime asset validation and reproducible run identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from .config import ExperimentConfig
from .io import atomic_write_json, canonical_hash, sha256_file


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    revision: str
    tokenizer_hash: str
    chat_template_hash: str | None


@dataclass(frozen=True)
class LockedRuntime:
    config: Any
    config_hash: str
    run_id: str


def validate_model_assets(path: str | Path) -> ResolvedModel:
    root = Path(path)
    config = root / "config.json"
    tokenizer_files = [item for item in root.glob("tokenizer*") if item.is_file()]
    weight_files = [item for item in root.glob("*.safetensors")] + [item for item in root.glob("pytorch_model*.bin")]
    if not root.is_dir() or not config.is_file() or not tokenizer_files or not weight_files:
        raise PreflightError(f"incomplete model snapshot: {root}")
    identity_files = sorted({config, *tokenizer_files, *weight_files})
    digest = hashlib.sha256("".join(f"{item.name}:{sha256_file(item)}\n" for item in identity_files).encode()).hexdigest()
    template = root / "chat_template.jinja"
    return ResolvedModel(root, f"local-sha256:{digest}", canonical_hash({item.name: sha256_file(item) for item in tokenizer_files}), sha256_file(template) if template.is_file() else None)


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def lock_runtime_config(config: Any, *, output_root: str | Path, source_hashes: Mapping[str, str]) -> LockedRuntime:
    """Lock a configuration object that exposes the base run contract.

    The independent H1-v2 protocol deliberately has a separate top-level
    configuration, while retaining the same immutable run metadata contract.
    """
    root = Path(output_root)
    payload = config.model_dump(mode="json")
    config_hash = canonical_hash(payload)
    run_id = f"run:{canonical_hash({'config_hash': config_hash, 'sources': dict(sorted(source_hashes.items()))})[:20]}"
    run = getattr(config, "run", None)
    if run is None:
        run = getattr(getattr(config, "base", None), "run", None)
    locked_name = getattr(run, "locked_config_name", None)
    if not isinstance(locked_name, str) or not locked_name:
        raise PreflightError("configuration has no locked run metadata")
    locked_path = root / locked_name
    if locked_path.exists() and json.loads(locked_path.read_text(encoding="utf-8")) != payload:
        raise PreflightError("output root already has a different locked config")
    atomic_write_json(locked_path, payload)
    atomic_write_json(root / "run_manifest.json", {"run_id": run_id, "config_hash": config_hash, "source_hashes": dict(sorted(source_hashes.items())), "git_revision": _git(["git", "rev-parse", "HEAD"]), "git_status_hash": canonical_hash(_git(["git", "status", "--porcelain"]))})
    return LockedRuntime(config, config_hash, run_id)
