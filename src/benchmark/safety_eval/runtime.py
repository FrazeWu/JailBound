"""Offline-only runtime asset validation and reproducible run identities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping

import torch

from .config import ExperimentConfig
from .io import atomic_write_json, canonical_hash, canonical_json, sha256_file
from .runner import OptimizationJob, stable_state_id
from .schema import (
    OptimizationRecord,
    RecordStatus,
    V2BenchmarkExample,
    V2MaterializationRecord,
    V2JudgmentRecord,
    V2ResponseRecord,
    token_ids_sha256,
)
from .io import read_jsonl


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
    # A mixed ledger can otherwise evade the old first-row check and be used by
    # resume code after a v2 run has begun.  Validate every row in every v2
    # execution ledger before selecting a single requested source.
    _validate_v2_jsonl_rows(
        root / "manifests" / "v2",
        "controlled_*.jsonl",
        V2BenchmarkExample,
        label="manifest",
    )
    _validate_v2_jsonl_rows(
        root / "optimization",
        "*/*/records.jsonl",
        OptimizationRecord,
        label="optimization",
    )
    _validate_v2_jsonl_rows(
        root / "optimization",
        "*/*/materialization.jsonl",
        V2MaterializationRecord,
        label="materialization",
    )
    _validate_v2_jsonl_rows(
        root / "responses",
        "*/*/*/records.jsonl",
        V2ResponseRecord,
        label="response",
    )
    _validate_v2_jsonl_rows(
        root / "judgments",
        "*/*/*/*/records.jsonl",
        V2JudgmentRecord,
        label="judgment",
    )


def _validate_v2_jsonl_rows(
    directory: Path,
    pattern: str,
    model_type: type[object],
    *,
    label: str,
) -> None:
    """Reject malformed and non-v2 rows in every structured v2 artifact."""
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        try:
            rows = read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise PreflightError(
                f"invalid {label} artifact in reviewer-v2 output root: {path}"
            ) from error
        for row in rows:
            if row.get("schema_version") != "reviewer_eval.v2":
                raise PreflightError("schema-v1 artifact exists in reviewer-v2 output root")
            try:
                # Every accepted row must be its declared structured record,
                # rather than merely carrying a v2-looking string field.
                getattr(model_type, "model_validate")(row)
            except (TypeError, ValueError) as error:
                if label == "response":
                    raise PreflightError(
                        f"legacy or invalid v2 response provenance in {path}"
                    ) from error
                raise PreflightError(
                    f"invalid {label} artifact in reviewer-v2 output root: {path}"
                ) from error


def validate_v2_provenance_ledgers(output_root: str | Path) -> None:
    """Validate v2 state, materialization, and response identity before resume."""
    root = Path(output_root)
    validate_v2_output_root(root)
    materializations: dict[str, V2MaterializationRecord] = {}
    optimizations: list[OptimizationRecord] = []
    response_execution_identities: set[tuple[object, ...]] = set()
    judgment_identities: set[tuple[object, ...]] = set()
    examples = _authoritative_v2_examples(root)

    for path in (root / "optimization").glob("*/*/records.jsonl") if (root / "optimization").exists() else ():
        for row in read_jsonl(path):
            try:
                record = OptimizationRecord.model_validate(row)
            except ValueError as error:
                raise PreflightError(f"legacy or invalid v2 optimization provenance in {path}") from error
            if record.status is RecordStatus.complete and record.state_sha256 is None:
                raise PreflightError(f"legacy v2 optimization provenance is missing in {path}")
            if record.status is RecordStatus.complete:
                optimizations.append(record)

    for path in (root / "optimization").glob("*/*/materialization.jsonl") if (root / "optimization").exists() else ():
        for row in read_jsonl(path):
            try:
                record = V2MaterializationRecord.model_validate(row)
            except ValueError as error:
                raise PreflightError(f"legacy or invalid v2 materialization provenance in {path}") from error
            existing = materializations.get(record.materialization_sha256)
            if existing is not None and existing != record:
                raise PreflightError("conflicting v2 materialization hash across ledgers")
            materializations[record.materialization_sha256] = record

    for materialization in materializations.values():
        _validate_v2_materialization_state_chain(
            root, materialization, optimizations, examples
        )

    for path in (root / "responses").glob("*/*/*/records.jsonl") if (root / "responses").exists() else ():
        for row in read_jsonl(path):
            try:
                response = V2ResponseRecord.model_validate(row)
            except ValueError as error:
                raise PreflightError(f"legacy or invalid v2 response provenance in {path}") from error
            execution_identity = (
                response.run_id,
                response.config_hash,
                response.sample_id,
                response.source,
                response.method,
                response.checkpoint,
                response.state_step,
                response.branch,
                response.transport,
                response.target_key,
                response.materialization_sha256,
                response.target_revision,
                response.target_tokenizer_sha256,
            )
            if execution_identity in response_execution_identities:
                raise PreflightError("duplicate v2 target execution response")
            response_execution_identities.add(execution_identity)
            materialization = materializations.get(response.materialization_sha256)
            if materialization is None:
                raise PreflightError(f"dangling v2 response materialization in {path}")
            _validate_v2_response_materialization_identity(response, materialization, path, root)
            expected_hash = token_ids_sha256(materialization.complete_token_ids)
            if (
                response.executed_token_ids_sha256 != expected_hash
                or response.prompt_hash != expected_hash
            ):
                raise PreflightError(f"v2 response token hash does not match materialization in {path}")
            if response.target_tokenizer_sha256 != materialization.surrogate_tokenizer_sha256:
                raise PreflightError(f"v2 response target tokenizer does not match materialization in {path}")

    for path in (root / "judgments").glob("*/*/*/*/records.jsonl") if (root / "judgments").exists() else ():
        for row in read_jsonl(path):
            try:
                judgment = V2JudgmentRecord.model_validate(row)
            except ValueError as error:
                raise PreflightError(f"legacy or invalid v2 judgment provenance in {path}") from error
            judgment_identity = (
                judgment.run_id,
                judgment.config_hash,
                judgment.sample_id,
                judgment.source,
                judgment.method,
                judgment.checkpoint,
                judgment.state_step,
                judgment.branch,
                judgment.transport,
                judgment.target_key,
                judgment.materialization_sha256,
                judgment.target_revision,
                judgment.target_tokenizer_sha256,
                judgment.judge_key,
                judgment.judge_revision,
                judgment.threshold,
            )
            if judgment_identity in judgment_identities:
                raise PreflightError("duplicate v2 judgment identity")
            judgment_identities.add(judgment_identity)
            _validate_v2_judgment_response_identity(judgment, materializations, root, path)


def _validate_v2_response_materialization_identity(
    response: V2ResponseRecord,
    materialization: V2MaterializationRecord,
    path: Path,
    output_root: Path,
) -> None:
    """Require execution metadata and ledger placement to name its projection."""
    response_identity = (
        response.run_id,
        response.config_hash,
        response.sample_id,
        response.source,
        response.method,
        response.checkpoint,
        response.state_step,
        response.branch,
        response.transport,
    )
    materialization_identity = (
        materialization.run_id,
        materialization.config_hash,
        materialization.sample_id,
        materialization.source,
        materialization.method,
        materialization.step,
        materialization.step,
        materialization.branch,
        materialization.transport,
    )
    if response_identity != materialization_identity:
        raise PreflightError(
            f"v2 response identity does not match materialization in {path}"
        )
    try:
        relative_parts = path.relative_to(output_root / "responses").parts
    except ValueError as error:
        raise PreflightError("v2 response ledger is outside the response root") from error
    if relative_parts != (
        response.target_key,
        response.source,
        response.method,
        "records.jsonl",
    ):
        if len(relative_parts) != 4:
            raise PreflightError(f"v2 response ledger has an invalid layout: {path}")
        for field, actual, expected in zip(
            ("target", "source", "method", "filename"),
            relative_parts,
            (response.target_key, response.source, response.method, "records.jsonl"),
            strict=True,
        ):
            if actual != expected:
                raise PreflightError(
                    f"v2 response ledger {field} partition does not match response in {path}"
                )
        raise PreflightError(f"v2 response ledger has an invalid layout: {path}")


def _validate_v2_judgment_response_identity(
    judgment: V2JudgmentRecord,
    materializations: Mapping[str, V2MaterializationRecord],
    output_root: Path,
    path: Path,
) -> None:
    """Require a v2 judgment to name exactly one persisted target execution."""
    materialization = materializations.get(judgment.materialization_sha256)
    if materialization is None:
        raise PreflightError(f"dangling v2 judgment materialization in {path}")
    response_path = (
        output_root / "responses" / judgment.target_key / judgment.source
        / judgment.method / "records.jsonl"
    )
    matches: list[V2ResponseRecord] = []
    for row in read_jsonl(response_path):
        try:
            response = V2ResponseRecord.model_validate(row)
        except ValueError as error:
            raise PreflightError(f"legacy or invalid v2 response provenance in {response_path}") from error
        if (
            response.materialization_sha256 == judgment.materialization_sha256
            and response.target_revision == judgment.target_revision
            and response.target_tokenizer_sha256 == judgment.target_tokenizer_sha256
            and response.run_id == judgment.run_id
            and response.config_hash == judgment.config_hash
            and response.sample_id == judgment.sample_id
            and response.source == judgment.source
            and response.method == judgment.method
            and response.checkpoint == judgment.checkpoint
            and response.state_step == judgment.state_step
            and response.branch == judgment.branch
            and response.transport == judgment.transport
            and response.target_key == judgment.target_key
        ):
            matches.append(response)
    if len(matches) != 1:
        raise PreflightError(f"v2 judgment does not join exactly one response in {path}")
    _validate_v2_response_materialization_identity(matches[0], materialization, response_path, output_root)
    try:
        parts = path.relative_to(output_root / "judgments").parts
    except ValueError as error:
        raise PreflightError("v2 judgment ledger is outside the judgment root") from error
    expected = (judgment.judge_key, judgment.target_key, judgment.source, judgment.method, "records.jsonl")
    if parts != expected:
        raise PreflightError(f"v2 judgment ledger placement does not match judgment in {path}")


def _validate_v2_materialization_state_chain(
    output_root: Path,
    materialization: V2MaterializationRecord,
    optimizations: list[OptimizationRecord],
    examples: Mapping[tuple[str, str], V2BenchmarkExample],
) -> None:
    """Join a projection ledger row to the exact persisted optimizer state.

    A materialization digest only commits to its own payload.  This check makes
    its referenced state digest meaningful by requiring the unique completed
    optimizer record that produced it, then rechecking the state contracts
    needed for token reconstruction before any response can be resumed.
    """
    matches = [
        record
        for record in optimizations
        if record.run_id == materialization.run_id
        and record.config_hash == materialization.config_hash
        and record.sample_id == materialization.sample_id
        and record.source == materialization.source
        and record.method == materialization.method
        and record.checkpoint == materialization.step
        and record.state_sha256 == materialization.state_sha256
        and record.representation.rsplit(":", 1)[-1] == materialization.branch
    ]
    if len(matches) != 1:
        raise PreflightError("v2 materialization has no unique matching optimization record")
    optimization = matches[0]
    example = examples.get((materialization.source, materialization.sample_id))
    if example is None:
        raise PreflightError("v2 materialization has no matching authoritative manifest example")
    if not optimization.state_path:
        raise PreflightError("v2 materialization state path is missing")
    state_path = _validate_v2_optimization_state_path(output_root, optimization)
    if not state_path.is_file():
        raise PreflightError("v2 materialization state file is missing")
    try:
        state_sha256 = sha256_file(state_path)
    except OSError as error:
        raise PreflightError("v2 materialization state file cannot be read") from error
    if state_sha256 != materialization.state_sha256:
        raise PreflightError("v2 materialization state sha256 does not match state file")
    try:
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise PreflightError("v2 materialization state cannot be decoded") from error
    if not isinstance(payload, Mapping):
        raise PreflightError("v2 materialization state must be a mapping")
    if payload.get("input_embedding_sha256") != materialization.surrogate_embedding_sha256:
        raise PreflightError("v2 materialization embedding contract does not match state")
    if payload.get("tokenizer_revision") != materialization.surrogate_tokenizer_sha256:
        raise PreflightError("v2 materialization tokenizer contract does not match state")
    expected_span_hashes = tuple(
        canonical_hash(span.model_dump(mode="json")) for span in example.editable_spans
    )
    if payload.get("editable_span_hashes") != expected_span_hashes:
        raise PreflightError("v2 materialization editable-span contract does not match manifest")
    base_token_ids = payload.get("base_token_ids")
    editable_positions = payload.get("editable_positions")
    if (
        not isinstance(base_token_ids, torch.Tensor)
        or base_token_ids.ndim != 2
        or base_token_ids.shape[0] != 1
        or not _is_integral_token_tensor(base_token_ids)
        or tuple(
        int(value) for value in base_token_ids.detach().cpu().reshape(-1).tolist()
        ) != materialization.original_token_ids
    ):
        raise PreflightError("v2 materialization base-token contract does not match state")
    if (
        not isinstance(editable_positions, torch.Tensor)
        or editable_positions.ndim != 1
        or not _is_integral_token_tensor(editable_positions)
        or tuple(
        int(value) for value in editable_positions.detach().cpu().reshape(-1).tolist()
        ) != materialization.editable_positions
    ):
        raise PreflightError("v2 materialization editable-position contract does not match state")


def _authoritative_v2_examples(root: Path) -> dict[tuple[str, str], V2BenchmarkExample]:
    examples: dict[tuple[str, str], V2BenchmarkExample] = {}
    for path in (root / "manifests" / "v2").glob("controlled_*.jsonl"):
        for row in read_jsonl(path):
            try:
                example = V2BenchmarkExample.model_validate(row)
            except ValueError as error:
                raise PreflightError(f"invalid v2 manifest provenance in {path}") from error
            identity = (example.source, example.example_id)
            if examples.get(identity) not in (None, example):
                raise PreflightError("conflicting authoritative v2 manifest example")
            examples[identity] = example
    return examples


def _validate_v2_optimization_state_path(
    output_root: Path, optimization: OptimizationRecord
) -> Path:
    """Require the runner-owned, non-symlink checkpoint path for a v2 record."""
    if not optimization.state_path:
        raise PreflightError("v2 materialization state path is missing")
    expected = (
        output_root
        / "optimization"
        / optimization.source
        / optimization.method
        / "states"
        / f"{stable_state_id(OptimizationJob(optimization.source, optimization.method, optimization.cell_id, optimization.sample_id, optimization.random_seed), optimization.checkpoint)}.pt"
    )
    supplied = Path(optimization.state_path)
    # The spelling of the generated checkpoint path is part of its provenance.
    # Checking only the file and ``states`` boundary permits a symlink at an
    # earlier controlled ancestor (optimization/source/method) to redirect the
    # complete checkpoint tree while retaining the expected resolved path.
    expected_ancestors = (
        output_root / "optimization",
        output_root / "optimization" / optimization.source,
        output_root / "optimization" / optimization.source / optimization.method,
        expected.parent,
        expected,
    )
    if any(path.is_symlink() for path in expected_ancestors):
        raise PreflightError("v2 optimization state path cannot be a symlink")
    try:
        if supplied.resolve(strict=False) != expected.resolve(strict=False):
            raise PreflightError("v2 optimization state path is outside its runner-managed location")
    except OSError as error:
        raise PreflightError("v2 optimization state path cannot be resolved") from error
    if not supplied.is_file():
        raise PreflightError("v2 materialization state file is missing")
    return supplied


def _is_integral_token_tensor(value: torch.Tensor) -> bool:
    return not value.dtype.is_floating_point and not value.dtype.is_complex and value.dtype != torch.bool


def require_v2_materialization_ledger_membership(
    output_root: str | Path, record: V2MaterializationRecord
) -> None:
    """Require the exact v2 record to be present in its authoritative ledger.

    A valid schema object supplied by a caller is insufficient authority to
    execute projected IDs: it must be byte-for-byte equivalent in canonical
    JSON terms to the materialization ledger generated from the checkpoint.
    """
    path = Path(output_root) / "optimization" / record.source / record.method / "materialization.jsonl"
    matches = [
        row for row in read_jsonl(path)
        if row.get("schema_version") == "reviewer_eval.v2"
        and row.get("materialization_sha256") == record.materialization_sha256
    ]
    if not matches:
        raise PreflightError("v2 materialization is absent from its authoritative ledger")
    expected = record.model_dump(mode="json")
    if any(canonical_json(row) != canonical_json(expected) for row in matches):
        raise PreflightError("v2 materialization ledger payload does not match supplied record")
    if len(matches) != 1:
        raise PreflightError("duplicate v2 materialization hash in authoritative ledger")


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
