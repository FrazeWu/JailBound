"""Offline-only runtime asset validation and reproducible run identities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping

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
    config: ExperimentConfig
    config_hash: str
    run_id: str


_TOKENIZER_ASSET_NAMES = frozenset(
    {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
    }
)
_SNAPSHOT_METADATA_DIRECTORIES = frozenset({".cache", ".git", "__pycache__"})


def validate_v2_output_root(output_root: str | Path) -> None:
    """Reject legacy artifacts before a reviewer-v2 stage touches an output root."""
    root = Path(output_root)
    manifests = root / "manifests"
    legacy = tuple(manifests.glob("controlled_*.jsonl")) + tuple(
        manifests.glob("controlled_*.header.json")
    )
    if legacy:
        raise PreflightError("schema-v1 artifact exists in reviewer-v2 output root")
    for path in (root / "optimization").glob("*/*/records.jsonl") if (root / "optimization").exists() else ():
        try:
            first = next(iter(path.open(encoding="utf-8"))).strip()
            payload = json.loads(first) if first else {}
        except (OSError, StopIteration, json.JSONDecodeError):
            raise PreflightError("invalid optimization artifact in reviewer-v2 output root")
        if payload.get("schema_version") != "reviewer_eval.v2":
            raise PreflightError("schema-v1 artifact exists in reviewer-v2 output root")


def validate_model_assets(path: str | Path) -> ResolvedModel:
    root = Path(path)
    config = root / "config.json"
    snapshot_files = sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file()
            and not any(
                part in _SNAPSHOT_METADATA_DIRECTORIES
                for part in item.relative_to(root).parts
            )
        ),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    tokenizer_files = [
        item
        for item in snapshot_files
        if item.name.startswith("tokenizer") or item.name in _TOKENIZER_ASSET_NAMES
    ]
    weight_files = [
        item
        for item in snapshot_files
        if item.name.endswith(".safetensors")
        or (item.name.startswith("pytorch_model") and item.name.endswith(".bin"))
    ]
    if (
        not root.is_dir()
        or config not in snapshot_files
        or not tokenizer_files
        or not weight_files
    ):
        raise PreflightError(f"incomplete model snapshot: {root}")
    snapshot_hashes = {
        item.relative_to(root).as_posix(): sha256_file(item)
        for item in snapshot_files
    }
    tokenizer_hashes = {
        item.relative_to(root).as_posix(): snapshot_hashes[
            item.relative_to(root).as_posix()
        ]
        for item in {config, *tokenizer_files}
    }
    digest = canonical_hash(snapshot_hashes)
    return ResolvedModel(
        root,
        f"local-sha256:{digest}",
        canonical_hash(tokenizer_hashes),
        snapshot_hashes.get("chat_template.jinja"),
    )


def _git(command: list[str], *, repository_root: str | Path) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=Path(repository_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_provenance(repository_root: str | Path) -> dict[str, str]:
    return {
        "git_revision": _git(
            ["git", "rev-parse", "HEAD"], repository_root=repository_root
        ),
        "git_status_hash": canonical_hash(
            _git(
                ["git", "status", "--porcelain"],
                repository_root=repository_root,
            )
        ),
    }


def lock_runtime_config(
    config: ExperimentConfig,
    *,
    output_root: str | Path,
    source_hashes: Mapping[str, str],
    repository_root: str | Path | None = None,
) -> LockedRuntime:
    root = Path(output_root)
    payload = config.model_dump(mode="json")
    config_hash = canonical_hash(payload)
    run_id = f"run:{canonical_hash({'config_hash': config_hash, 'sources': dict(sorted(source_hashes.items()))})[:20]}"
    locked_path = root / config.run.locked_config_name
    if locked_path.exists() and json.loads(locked_path.read_text(encoding="utf-8")) != payload:
        raise PreflightError("output root already has a different locked config")
    atomic_write_json(locked_path, payload)
    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[3]
    provenance = git_provenance(repository_root)
    atomic_write_json(
        root / "run_manifest.json",
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "source_hashes": dict(sorted(source_hashes.items())),
            **provenance,
        },
    )
    return LockedRuntime(config, config_hash, run_id)
