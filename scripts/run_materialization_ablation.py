"""Run a same-state continuous/materialized ablation on locked Qwen2.5-7B artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import torch
import torch.nn.functional as functional
from omegaconf import OmegaConf

from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.generation import generate_from_embeddings, generate_one
from benchmark.safety_eval.io import JsonlLedger, atomic_write_json, canonical_hash, read_jsonl
from benchmark.safety_eval.materialization_ablation import Branch, MaterializationPair
from benchmark.safety_eval.runtime import validate_model_assets


ROOT = Path(__file__).resolve().parents[1]
QWEN_7B_KEY = "qwen2_5_7b"
QWEN_7B_REPO = "Qwen/Qwen2.5-7B-Instruct"


class AblationUnit(NamedTuple):
    source: str
    sample_id: str
    branch: Branch
    method: str
    checkpoint: int
    attack_text: str
    reference_intent: str
    initial_state_path: Path | None
    final_state_path: Path | None
    final_state_sha256: str | None
    config_hash: str | None
    error: str | None


class LockedAblationInputs(NamedTuple):
    source_root: Path
    output_root: Path
    sources: tuple[str, ...]
    expected_samples_per_source: int
    checkpoint: int
    branch_methods: dict[Branch, str]
    model_key: str
    model_repo_id: str
    model_path: Path
    model_revision: str
    hidden_size: int
    attention_implementation: str
    max_new_tokens: int
    units: tuple[AblationUnit, ...]
    config_payload: dict[str, object]


class ProjectionAudit(NamedTuple):
    prefix_token_ids: tuple[int, ...]
    seed_token_ids: tuple[int, ...]
    cosines: tuple[float, ...]


class ContinuousChatInput(NamedTuple):
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    content_token_ids: tuple[int, ...]
    content_start: int


class RoundTripAudit(NamedTuple):
    materialized_text: str
    retokenized_token_ids: tuple[int, ...]
    roundtrip_exact_match: bool


def _absolute(path: Path, *, relative_to: Path = ROOT) -> Path:
    return path if path.is_absolute() else relative_to / path


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _non_empty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def sha256_file(path: Path) -> str:
    """Hash one immutable artifact without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_tensors(path: Path, *, hidden_size: int) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"state payload is not a mapping: {path}")
    result: dict[str, torch.Tensor] = {}
    for name in ("z", "u"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(f"state has invalid {name} tensor: {path}")
        if value.shape[-1] != hidden_size:
            raise ValueError(f"state hidden size differs from configured Qwen2.5-7B hidden size: {path}")
        result[name] = value
    return result


def _terminal_rows(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for row in read_jsonl(path):
        sample_id, checkpoint = row.get("sample_id"), row.get("checkpoint")
        if not isinstance(sample_id, str) or isinstance(checkpoint, bool) or not isinstance(checkpoint, int):
            raise ValueError(f"optimization ledger has invalid identity: {path}")
        key = sample_id, checkpoint
        if key in indexed:
            raise ValueError(f"duplicate optimization record: {key}")
        indexed[key] = row
    return indexed


def _state_path(row: dict[str, object] | None, *, hidden_size: int) -> tuple[Path | None, str | None]:
    if row is None or row.get("status") != "complete":
        return None, "optimization checkpoint is missing or failed"
    raw_path = row.get("state_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None, "optimization checkpoint has no state path"
    path = _absolute(Path(raw_path))
    if not path.is_file():
        return None, f"state file is missing: {path}"
    _state_tensors(path, hidden_size=hidden_size)
    return path, None


def load_locked_inputs(config_path: Path) -> LockedAblationInputs:
    """Resolve and validate immutable sample/state pairs without loading a model."""
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    config = _mapping(payload, label="ablation config")
    source_root = _absolute(Path(_non_empty_string(config.get("source_root"), label="source_root")))
    output_root = _absolute(Path(_non_empty_string(config.get("output_root"), label="output_root")))
    if source_root.resolve() == output_root.resolve():
        raise ValueError("ablation output root must differ from the matched source root")
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources or not all(isinstance(value, str) and value for value in raw_sources):
        raise ValueError("sources must be a non-empty string list")
    sources = tuple(raw_sources)
    if len(set(sources)) != len(sources):
        raise ValueError("sources must be unique")
    expected_samples = _positive_int(config.get("expected_samples_per_source"), label="expected samples")
    checkpoint = _positive_int(config.get("checkpoint"), label="checkpoint")
    if checkpoint != 100:
        raise ValueError("Qwen2.5-7B matched ablation requires checkpoint 100")

    raw_branches = _mapping(config.get("branch_methods"), label="branch_methods")
    branch_methods = {Branch(key): _non_empty_string(value, label=f"method for {key}") for key, value in raw_branches.items()}
    expected_branch_methods = {
        Branch.high_value: "jailbound_o_minus",
        Branch.safety_sensitivity: "jailbound_o_plus",
    }
    if branch_methods != expected_branch_methods:
        raise ValueError("branch_methods must use the exact branch mapping")

    model = _mapping(config.get("model"), label="model")
    model_key = _non_empty_string(model.get("key"), label="model key")
    model_repo_id = _non_empty_string(model.get("repo_id"), label="model repo_id")
    if model_key != QWEN_7B_KEY or model_repo_id != QWEN_7B_REPO:
        raise ValueError("materialization ablation is locked to Qwen2.5-7B")
    model_path = _absolute(Path(_non_empty_string(model.get("local_path"), label="model local_path")))
    model_revision = _non_empty_string(model.get("revision"), label="model revision")
    hidden_size = _positive_int(model.get("hidden_size"), label="model hidden size")
    attention = _non_empty_string(model.get("attention_implementation"), label="attention implementation")
    max_new_tokens = _positive_int(config.get("max_new_tokens"), label="max_new_tokens")
    if max_new_tokens != 512:
        raise ValueError("materialization ablation requires max_new_tokens=512")

    locked_path = source_root / "locked_config.json"
    locked = _mapping(json.loads(locked_path.read_text(encoding="utf-8")), label="matched locked config")
    locked_model = _mapping(_mapping(locked.get("models"), label="matched models").get("surrogate"), label="matched surrogate")
    if locked_model.get("key") != model_key or locked_model.get("repo_id") != model_repo_id:
        raise ValueError("matched artifacts are not from the configured Qwen2.5-7B model")
    locked_optimization = _mapping(locked.get("optimization"), label="matched optimization")
    if locked_optimization.get("update_budget") != checkpoint:
        raise ValueError("matched update budget differs from the ablation checkpoint")

    units: list[AblationUnit] = []
    observed_config_hashes: set[str] = set()
    for source in sources:
        manifest_path = source_root / "manifests" / f"controlled_{source}.jsonl"
        examples = read_jsonl(manifest_path)
        if len(examples) != expected_samples:
            raise ValueError(f"matched sample count differs for {source}")
        example_by_id: dict[str, dict[str, object]] = {}
        for example in examples:
            sample_id = _non_empty_string(example.get("example_id"), label="example_id")
            if sample_id in example_by_id:
                raise ValueError(f"duplicate manifest sample: {sample_id}")
            if example.get("source") != source:
                raise ValueError("manifest source does not match its partition")
            example_by_id[sample_id] = example
        for branch, method in branch_methods.items():
            rows = _terminal_rows(source_root / "optimization" / source / method / "records.jsonl")
            for row in rows.values():
                config_hash = row.get("config_hash")
                if isinstance(config_hash, str):
                    observed_config_hashes.add(config_hash)
            for sample_id in sorted(example_by_id):
                example = example_by_id[sample_id]
                initial_path, initial_error = _state_path(rows.get((sample_id, 0)), hidden_size=hidden_size)
                final_row = rows.get((sample_id, checkpoint))
                final_path, final_error = _state_path(final_row, hidden_size=hidden_size)
                error = initial_error or final_error
                final_hash = sha256_file(final_path) if final_path is not None else None
                config_hash = final_row.get("config_hash") if final_row else None
                units.append(AblationUnit(
                    source=source,
                    sample_id=sample_id,
                    branch=branch,
                    method=method,
                    checkpoint=checkpoint,
                    attack_text=_non_empty_string(example.get("attack_text"), label="attack_text"),
                    reference_intent=_non_empty_string(example.get("intent"), label="intent"),
                    initial_state_path=initial_path,
                    final_state_path=final_path,
                    final_state_sha256=final_hash,
                    config_hash=config_hash if isinstance(config_hash, str) else None,
                    error=error,
                ))
    if len(observed_config_hashes) != 1:
        raise ValueError("matched optimization records have mixed config hashes")
    return LockedAblationInputs(
        source_root=source_root,
        output_root=output_root,
        sources=sources,
        expected_samples_per_source=expected_samples,
        checkpoint=checkpoint,
        branch_methods=branch_methods,
        model_key=model_key,
        model_repo_id=model_repo_id,
        model_path=model_path,
        model_revision=model_revision,
        hidden_size=hidden_size,
        attention_implementation=attention,
        max_new_tokens=max_new_tokens,
        units=tuple(units),
        config_payload=config,
    )


def _project_block(block: torch.Tensor, vocabulary: torch.Tensor, *, chunk_size: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if block.ndim != 3 or block.shape[0] != 1 or vocabulary.ndim != 2 or block.shape[-1] != vocabulary.shape[-1]:
        raise ValueError("projection tensors have incompatible shapes")
    if chunk_size < 1:
        raise ValueError("projection chunk size must be positive")
    queries = functional.normalize(block.detach().reshape(-1, block.shape[-1]).float(), dim=-1)
    best_scores = torch.full((queries.shape[0],), -torch.inf, device=queries.device)
    best_ids = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
    for start in range(0, vocabulary.shape[0], chunk_size):
        candidates = functional.normalize(vocabulary[start : start + chunk_size].detach().float(), dim=-1)
        scores = queries @ candidates.T
        chunk_scores, chunk_ids = scores.max(dim=-1)
        update = chunk_scores > best_scores
        best_scores = torch.where(update, chunk_scores, best_scores)
        best_ids = torch.where(update, chunk_ids + start, best_ids)
    return (
        tuple(int(value) for value in best_ids.detach().cpu().tolist()),
        tuple(float(value) for value in best_scores.detach().cpu().tolist()),
    )


def project_with_position_cosines(
    z: torch.Tensor,
    u: torch.Tensor,
    vocabulary: torch.Tensor,
    *,
    chunk_size: int = 2048,
) -> ProjectionAudit:
    """Project both editable blocks and retain one cosine per position."""
    prefix_ids, prefix_scores = _project_block(z, vocabulary, chunk_size=chunk_size)
    seed_ids, seed_scores = _project_block(u, vocabulary, chunk_size=chunk_size)
    return ProjectionAudit(prefix_ids, seed_ids, prefix_scores + seed_scores)


def _tensor_field(payload: object, field: str) -> torch.Tensor:
    value = payload.get(field) if isinstance(payload, dict) else getattr(payload, field, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"tokenizer did not return tensor {field}")
    return value


def build_continuous_chat_input(
    model: Any,
    tokenizer: Any,
    *,
    attack_text: str,
    z: torch.Tensor,
    u: torch.Tensor,
) -> ContinuousChatInput:
    """Insert continuous state at the user-content span of the standard chat template."""
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": attack_text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or rendered.count(attack_text) != 1:
        raise ValueError("chat template does not contain one exact user-content span")
    content_start_char = rendered.index(attack_text)
    content_end_char = content_start_char + len(attack_text)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    input_ids = _tensor_field(encoded, "input_ids")
    offsets = _tensor_field(encoded, "offset_mapping")
    if input_ids.ndim != 2 or offsets.shape != (*input_ids.shape, 2):
        raise ValueError("chat-template token offsets have invalid shape")
    selected = [
        index
        for index, (start, end) in enumerate(offsets[0].detach().cpu().tolist())
        if end > content_start_char and start < content_end_char
    ]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError("user-content token span is empty or non-contiguous")
    embedding_layer = model.get_input_embeddings()
    weight = getattr(embedding_layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError("model has no input embedding matrix")
    ids = input_ids.to(weight.device)
    with torch.no_grad():
        token_embeddings = embedding_layer(ids).detach()
    first, last = selected[0], selected[-1] + 1
    z = z.to(device=weight.device, dtype=token_embeddings.dtype)
    u = u.to(device=weight.device, dtype=token_embeddings.dtype)
    if z.ndim != 3 or u.ndim != 3 or z.shape[0] != 1 or u.shape[0] != 1 or z.shape[-1] != weight.shape[-1] or u.shape[-1] != weight.shape[-1]:
        raise ValueError("continuous state is incompatible with chat embeddings")
    inputs_embeds = torch.cat((token_embeddings[:, :first], z, token_embeddings[:, first:last], u, token_embeddings[:, last:]), dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    return ContinuousChatInput(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        content_token_ids=tuple(int(value) for value in ids[0, first:last].detach().cpu().tolist()),
        content_start=first,
    )


def materialized_roundtrip(tokenizer: Any, projected_token_ids: tuple[int, ...]) -> RoundTripAudit:
    """Decode the complete user sequence once and re-tokenize it without repair."""
    materialized_text = str(tokenizer.decode(list(projected_token_ids), skip_special_tokens=False))
    encoded = tokenizer(materialized_text, return_tensors="pt", add_special_tokens=False)
    token_ids = _tensor_field(encoded, "input_ids")
    retokenized = tuple(int(value) for value in token_ids.detach().reshape(-1).cpu().tolist())
    return RoundTripAudit(materialized_text, retokenized, retokenized == projected_token_ids)


def _failed_pair(locked: LockedAblationInputs, unit: AblationUnit, error: str) -> MaterializationPair:
    return MaterializationPair.model_validate({
        "source": unit.source,
        "sample_id": unit.sample_id,
        "branch": unit.branch.value,
        "optimization_checkpoint": unit.checkpoint,
        "state_sha256": unit.final_state_sha256,
        "model_key": locked.model_key,
        "model_revision": locked.model_revision,
        "initial_discrete_prompt": unit.attack_text,
        "reference_intent": unit.reference_intent,
        "continuous_response": "",
        "materialized_text": "",
        "materialized_response": "",
        "editable_projected_token_ids": [],
        "projected_token_ids": [],
        "retokenized_token_ids": [],
        "projection_cosines": [],
        "roundtrip_exact_match": False,
        "projected_length": 0,
        "retokenized_length": 0,
        "max_new_tokens": locked.max_new_tokens,
        "status": "failed",
        "error": error,
        "judgments": {},
    })


def run_unit(
    locked: LockedAblationInputs,
    unit: AblationUnit,
    *,
    model: Any,
    tokenizer: Any,
    projection_chunk_size: int = 2048,
) -> MaterializationPair:
    """Generate both conditions from one immutable state without semantic filtering."""
    if unit.error or unit.initial_state_path is None or unit.final_state_path is None:
        return _failed_pair(locked, unit, unit.error or "state is unavailable")
    try:
        initial = _state_tensors(unit.initial_state_path, hidden_size=locked.hidden_size)
        final = _state_tensors(unit.final_state_path, hidden_size=locked.hidden_size)
        embedding_weight = model.get_input_embeddings().weight.detach()
        continuous = build_continuous_chat_input(
            model, tokenizer, attack_text=unit.attack_text, z=final["z"], u=final["u"]
        )
        projection = project_with_position_cosines(
            final["z"].to(embedding_weight.device),
            final["u"].to(embedding_weight.device),
            embedding_weight,
            chunk_size=projection_chunk_size,
        )
        initial_projection = project_with_position_cosines(
            initial["z"].to(embedding_weight.device),
            initial["u"].to(embedding_weight.device),
            embedding_weight,
            chunk_size=projection_chunk_size,
        )
        projected_ids = projection.prefix_token_ids + continuous.content_token_ids + projection.seed_token_ids
        initial_ids = initial_projection.prefix_token_ids + continuous.content_token_ids + initial_projection.seed_token_ids
        initial_prompt = str(tokenizer.decode(list(initial_ids), skip_special_tokens=True))
        roundtrip = materialized_roundtrip(tokenizer, projected_ids)
        continuous_result = generate_from_embeddings(
            model,
            tokenizer,
            inputs_embeds=continuous.inputs_embeds,
            attention_mask=continuous.attention_mask,
            max_new_tokens=locked.max_new_tokens,
        )
        materialized_result = generate_one(
            model,
            tokenizer,
            "",
            roundtrip.materialized_text,
            locked.max_new_tokens,
        )
        return MaterializationPair.model_validate({
            "source": unit.source,
            "sample_id": unit.sample_id,
            "branch": unit.branch.value,
            "optimization_checkpoint": unit.checkpoint,
            "state_sha256": unit.final_state_sha256,
            "model_key": locked.model_key,
            "model_revision": locked.model_revision,
            "initial_discrete_prompt": initial_prompt,
            "reference_intent": unit.reference_intent,
            "continuous_response": continuous_result.response,
            "materialized_text": roundtrip.materialized_text,
            "materialized_response": materialized_result.response,
            "editable_projected_token_ids": projection.prefix_token_ids + projection.seed_token_ids,
            "projected_token_ids": projected_ids,
            "retokenized_token_ids": roundtrip.retokenized_token_ids,
            "projection_cosines": projection.cosines,
            "roundtrip_exact_match": roundtrip.roundtrip_exact_match,
            "projected_length": len(projected_ids),
            "retokenized_length": len(roundtrip.retokenized_token_ids),
            "max_new_tokens": locked.max_new_tokens,
            "status": "complete",
            "error": None,
            "judgments": {},
        })
    except (AttributeError, IndexError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        return _failed_pair(locked, unit, f"pair generation error: {type(error).__name__}: {error}")


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", action="append")
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--projection-chunk-size", type=int, default=2048)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    locked = load_locked_inputs(args.config)
    selected_sources = tuple(args.source or locked.sources)
    if set(selected_sources) - set(locked.sources):
        raise ValueError("requested source is not configured")
    units = tuple(unit for unit in locked.units if unit.source in selected_sources)
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise ValueError("max-pairs must be positive")
        units = units[: args.max_pairs]
    ready = sum(unit.error is None for unit in units)
    report = {
        "checkpoint": locked.checkpoint,
        "failed_units": len(units) - ready,
        "model_key": locked.model_key,
        "ready_units": ready,
        "sources": list(selected_sources),
        "total_units": len(units),
    }
    if args.dry_run:
        print(json.dumps(report, sort_keys=True))
        return 0

    resolved = validate_model_assets(locked.model_path)
    handle = load_local_qwen(resolved, attention_backend=locked.attention_implementation)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("Qwen2.5-7B model did not load")
        ledger = JsonlLedger(
            locked.output_root / "materialization_pairs.jsonl",
            key_fields=("source", "sample_id", "branch", "optimization_checkpoint"),
        )
        written = 0
        for unit in units:
            pair = run_unit(
                locked,
                unit,
                model=handle.model,
                tokenizer=handle.tokenizer,
                projection_chunk_size=args.projection_chunk_size,
            )
            written += int(ledger.append_once(pair.model_dump(mode="json")))
    finally:
        handle.close()
    atomic_write_json(locked.output_root / "run_manifest.json", {
        "schema_version": "materialization_ablation.v1",
        "config_hash": canonical_hash(locked.config_payload),
        "git_revision": _git_revision(),
        "source_root": str(locked.source_root),
        "model_key": locked.model_key,
        "model_repo_id": locked.model_repo_id,
        "model_revision": locked.model_revision,
        "checkpoint": locked.checkpoint,
        "sources": list(selected_sources),
        "total_units": len(units),
        "written_units": written,
    })
    print(json.dumps({**report, "written_units": written}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
