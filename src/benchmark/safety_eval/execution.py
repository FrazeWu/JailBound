"""Bounded, offline-preflight execution entrypoints for safety evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
import gc
from typing import Any

import torch

from .io import JsonlLedger, canonical_hash, read_jsonl
from .optimizers.base import BudgetLedger, CheckpointEmitter
from .optimizers.gbda import GBDAOptimizer
from .optimizers.gcg import GCGOptimizer
from .optimizers.jailbound import DualBranchOptimizer, InitOptimizer, build_jailbound_optimizer
from .optimizers.pez import PEZOptimizer
from .optimizers.random_mutation import RandomMutationOptimizer
from .objective import EditableState
from .runtime import ResolvedModel, validate_model_assets
from .runner import OptimizationJob, OptimizationRunner, OptimizationSnapshot
from .schema import (
    BenchmarkExample,
    ComputeCounters,
    FailureKind,
    ManifestHeader,
    OptimizationRecord,
    RecordStatus,
    stable_id,
)
from .transformer_objective import TransformerAttackObjective


class ExecutionError(RuntimeError):
    """A content-free preflight or execution setup failure."""


class ExecutionMode(StrEnum):
    dry_run = "dry-run"
    smoke = "smoke"


@dataclass(frozen=True)
class ExecutionRequest:
    output_root: Path
    locked_config_name: str
    schema_version: str
    local_model_path: Path
    source: str
    method: str
    checkpoints: tuple[int, ...]
    requested_limit: int
    seed: int
    shard_index: int = 0
    shard_count: int = 1
    requested_sample_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExecutionSummary:
    mode: ExecutionMode
    selected_records: int
    completed_records: int
    failed_records: int


_TENSOR_METHODS = frozenset(
    ("init", "random_mutation", "zol", "pez", "gbda", "gcg", "jailbound_o_minus", "jailbound_o_plus", "dual_branch")
)


def tensor_method_for_recovery(method: str) -> str:
    """Resolve the explicitly named recovery variants to their tensor method."""
    for suffix in (
        "_recovery_fd_sdpa",
        "_recovery_fd",
        "_recovery_rebalanced",
        "_recovery_checkpointed",
        "_recovery_eager_retry",
        "_recovery_sdpa",
        "_recovery_eager",
        "_recovery",
    ):
        if method.endswith(suffix):
            return method.removesuffix(suffix)
    return method


@dataclass(frozen=True)
class TensorOptimizationSettings:
    """Bounded tensor-optimization values copied from the locked configuration."""

    checkpoints: tuple[int, ...]
    update_budget: int
    dual_branch_updates: dict[str, int]
    candidate_cap: int
    prefix_tokens: int
    editable_seed_tokens: int
    learning_rate: float
    lambda_fol: float
    epsilon: float
    gamma_z: float
    gamma_u: float
    grad_clip: float
    answer_anchors: tuple[str, ...]
    refusal_anchors: tuple[str, ...]
    gbda_learning_rate: float | None = None
    gcg_search_width: int = 32
    finite_difference_fol: bool = False
    finite_difference_radius: float = 1e-3

    def __post_init__(self) -> None:
        if self.checkpoints != tuple(sorted(set(self.checkpoints))) or not self.checkpoints or self.checkpoints[0] != 0:
            raise ValueError("tensor checkpoints must be unique, ordered, and start at zero")
        if self.update_budget < 1 or self.candidate_cap < 1:
            raise ValueError("tensor optimization budgets must be positive")
        if self.prefix_tokens < 1 or self.editable_seed_tokens < 1 or self.gcg_search_width < 1:
            raise ValueError("editable token counts must be positive")
        if self.gbda_learning_rate is not None and self.gbda_learning_rate <= 0:
            raise ValueError("GBDA learning rate must be positive")
        if self.finite_difference_radius <= 0:
            raise ValueError("finite-difference FOL radius must be positive")
        if not self.answer_anchors or not self.refusal_anchors:
            raise ValueError("tensor optimization requires both anchor sets")


@dataclass
class LocalQwenHandle:
    """A locally loaded tokenizer/model pair that can be explicitly released."""

    tokenizer: Any | None
    model: Any | None
    cleanup: Callable[[], None] | None = None

    def close(self) -> None:
        model, self.model = self.model, None
        self.tokenizer = None
        if model is not None:
            move_to_cpu = getattr(model, "to", None)
            if callable(move_to_cpu):
                try:
                    move_to_cpu("cpu")
                except (RuntimeError, ValueError):
                    pass
        gc.collect()
        if self.cleanup is not None:
            self.cleanup()


def load_local_qwen(
    resolved: ResolvedModel,
    *,
    attention_backend: Literal["eager", "sdpa"] = "eager",
    activation_checkpointing: bool = False,
    device_map: Literal["auto", "balanced"] | Mapping[str, int | Literal["cpu"]] = "auto",
) -> LocalQwenHandle:
    """Load the validated local Qwen snapshot without network access."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if attention_backend not in {"eager", "sdpa"}:
        raise ValueError("attention backend must be eager or sdpa")
    if isinstance(device_map, str):
        if device_map not in {"auto", "balanced"}:
            raise ValueError("device map must be auto, balanced, or an explicit module map")
    elif not isinstance(device_map, Mapping) or not device_map or not all(
        isinstance(module, str) and (
            (isinstance(device, int) and device >= 0) or device == "cpu"
        )
        for module, device in device_map.items()
    ):
        raise ValueError("explicit device map must contain non-negative integer device IDs or cpu")

    tokenizer = AutoTokenizer.from_pretrained(resolved.path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved.path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attention_backend,
    ).eval()
    if activation_checkpointing:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise ExecutionError("local model does not support activation checkpointing")
        if getattr(getattr(model, "config", None), "attention_dropout", None) != 0.0:
            raise ExecutionError("activation checkpointing requires zero attention dropout")
        enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()

    def cleanup() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return LocalQwenHandle(tokenizer=tokenizer, model=model, cleanup=cleanup)


def _input_ids(encoded: object) -> object:
    if isinstance(encoded, dict):
        return encoded["input_ids"]
    return getattr(encoded, "input_ids")


def _token_count(token_ids: object) -> int:
    shape = getattr(token_ids, "shape", ())
    if len(shape) >= 2:
        return int(shape[-1])
    if hasattr(token_ids, "numel"):
        return int(token_ids.numel())
    return len(token_ids)  # type: ignore[arg-type]


def local_qwen_init_executor(
    loaded: LocalQwenHandle,
    record: BenchmarkExample,
    job: OptimizationJob,
    checkpoints: tuple[int, ...],
) -> Iterable[OptimizationSnapshot]:
    """Embed one record locally and return the immutable Init checkpoint only."""
    if job.method != "init":
        raise ExecutionError("local Qwen smoke executor supports only method 'init'")
    if checkpoints != (0,):
        raise ExecutionError("local Qwen smoke executor requires checkpoint 0 only")
    if loaded.tokenizer is None or loaded.model is None:
        raise ExecutionError("local Qwen handle is closed")

    encoded = loaded.tokenizer(record.attack_text, return_tensors="pt", add_special_tokens=True)
    token_ids = _input_ids(encoded)
    embedding_layer = loaded.model.get_input_embeddings()
    embedding_weight = getattr(embedding_layer, "weight", None)
    device = getattr(embedding_weight, "device", None)
    if device is not None and hasattr(token_ids, "to"):
        token_ids = token_ids.to(device)
    import torch

    with torch.inference_mode():
        embedding_layer(token_ids)
    return [
        OptimizationSnapshot(
            checkpoint=0,
            representation="init_token_embeddings",
            attack_loss=None,
            counters=ComputeCounters(prompt_tokens=_token_count(token_ids)),
        )
    ]


def _anchor_token_ids(tokenizer: Any, anchors: tuple[str, ...], device: torch.device) -> torch.Tensor:
    token_ids: set[int] = set()
    for anchor in anchors:
        encoded = tokenizer.encode(anchor, add_special_tokens=False)
        if encoded:
            token_ids.add(int(encoded[0]))
    if not token_ids:
        raise ExecutionError("local tensor executor could not encode anchor tokens")
    return torch.tensor(sorted(token_ids), dtype=torch.long, device=device)


def _embedding_device(model: Any) -> torch.device:
    embedding_layer = model.get_input_embeddings()
    weight = getattr(embedding_layer, "weight", None)
    device = getattr(weight, "device", None)
    return device if isinstance(device, torch.device) else torch.device("cpu")


def _initial_editable_token_ids(token_ids: torch.Tensor, settings: TensorOptimizationSettings) -> tuple[torch.Tensor, torch.Tensor]:
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
        raise ExecutionError("local tensor executor requires one non-empty tokenized input")
    z_ids = token_ids[:, :1].expand(-1, settings.prefix_tokens).clone()
    u_count = min(settings.editable_seed_tokens, token_ids.shape[1])
    return z_ids, token_ids[:, -u_count:].clone()


def _tensor_optimizer(method: str, settings: TensorOptimizationSettings) -> Any:
    if method == "init":
        return InitOptimizer()
    if method == "dual_branch":
        return DualBranchOptimizer(learning_rate=settings.learning_rate, max_grad_norm=settings.grad_clip)
    return build_jailbound_optimizer(
        method,
        learning_rate=settings.learning_rate,
        max_grad_norm=settings.grad_clip,
        finite_difference_fol=settings.finite_difference_fol,
        finite_difference_radius=settings.finite_difference_radius,
    )


def _vocabulary_embeddings(model: Any) -> torch.Tensor:
    weight = getattr(model.get_input_embeddings(), "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ExecutionError("local tensor executor requires a vocabulary embedding matrix")
    return weight.detach()


def _initial_state_for_method(
    method: str,
    *,
    objective: TransformerAttackObjective,
    z_ids: torch.Tensor,
    u_ids: torch.Tensor,
    embedding: torch.Tensor,
) -> Any:
    if method == "gbda":
        vocabulary_size = embedding.shape[0]
        z_logits = torch.full((*z_ids.shape, vocabulary_size), -8.0, device=z_ids.device, dtype=embedding.dtype)
        u_logits = torch.full((*u_ids.shape, vocabulary_size), -8.0, device=u_ids.device, dtype=embedding.dtype)
        z_logits.scatter_(-1, z_ids.unsqueeze(-1), 8.0)
        u_logits.scatter_(-1, u_ids.unsqueeze(-1), 8.0)
        from .objective import EditableState

        return EditableState(z_logits, u_logits, z_logits.detach().clone(), u_logits.detach().clone())
    if method == "gcg":
        from .objective import EditableState

        return EditableState(z_ids, u_ids, z_ids.detach().clone(), u_ids.detach().clone())
    return objective.build_editable_state(z_ids, u_ids)


def _baseline_optimizer(method: str, settings: TensorOptimizationSettings, embedding: torch.Tensor) -> Any:
    if method == "pez":
        return PEZOptimizer(embedding, learning_rate=settings.learning_rate, max_grad_norm=settings.grad_clip)
    if method == "gbda":
        return GBDAOptimizer(
            embedding,
            learning_rate=settings.gbda_learning_rate or settings.learning_rate,
            max_grad_norm=settings.grad_clip,
        )
    if method == "gcg":
        return GCGOptimizer(embedding, search_width=settings.gcg_search_width, top_k=settings.gcg_search_width)
    return build_jailbound_optimizer(method, learning_rate=settings.learning_rate, max_grad_norm=settings.grad_clip)


def _checkpoint_state(snapshot: Any) -> dict[str, torch.Tensor]:
    state = getattr(snapshot, "state", None)
    if state is None:
        raise ExecutionError("optimizer checkpoint did not include an editable state")
    z_token_ids = getattr(snapshot, "z_token_ids", torch.empty((0, 0), dtype=torch.long))
    u_token_ids = getattr(snapshot, "u_token_ids", torch.empty((0, 0), dtype=torch.long))
    return {
        "z": state.z.detach().to(device="cpu"),
        "u": state.u.detach().to(device="cpu"),
        "z_token_ids": z_token_ids.detach().to(device="cpu", dtype=torch.long),
        "u_token_ids": u_token_ids.detach().to(device="cpu", dtype=torch.long),
    }


def _tensor_ledger(method: str, settings: TensorOptimizationSettings) -> BudgetLedger:
    branch_limits = dict(settings.dual_branch_updates) if method == "dual_branch" else {}
    return BudgetLedger(
        update_limit=settings.update_budget,
        candidate_limit=settings.candidate_cap,
        branch_limits=branch_limits,
    )


def _clone_state(state: EditableState) -> EditableState:
    return EditableState(
        z=state.z.detach().clone(),
        u=state.u.detach().clone(),
        z0=state.z0.detach().clone(),
        u0=state.u0.detach().clone(),
    )


def _random_mutation_snapshots(
    *,
    objective: TransformerAttackObjective,
    initial_state: EditableState,
    settings: TensorOptimizationSettings,
    seed: int,
    prompt_tokens: int,
) -> list[OptimizationSnapshot]:
    """Run a deterministic, semantic-proxy constrained vector mutation baseline."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    reference = torch.cat((initial_state.z.detach().float().reshape(-1), initial_state.u.detach().float().reshape(-1)))

    def rewriter(candidate: EditableState) -> EditableState:
        proposal = _clone_state(candidate)
        block = proposal.z if int(torch.randint(0, 2, (), generator=generator).item()) == 0 else proposal.u
        position = int(torch.randint(0, block.shape[1], (), generator=generator).item())
        noise = torch.randn(block[:, position, :].shape, dtype=torch.float32, generator=generator).to(
            device=block.device, dtype=block.dtype
        )
        block[:, position, :] = block[:, position, :] + 1e-4 * torch.nn.functional.normalize(noise, dim=-1)
        return proposal

    def semantic_acceptor(candidate: EditableState) -> bool:
        vector = torch.cat((candidate.z.detach().float().reshape(-1), candidate.u.detach().float().reshape(-1)))
        similarity = torch.nn.functional.cosine_similarity(reference, vector, dim=0)
        return bool(similarity >= 0.999)

    def score(candidate: EditableState) -> float:
        return float(objective.evaluate(candidate, include_fol=False).attack_loss.detach().cpu())

    snapshots = RandomMutationOptimizer().run(
        initial_candidate=_clone_state(initial_state),
        objective_score=score,
        rewriter=rewriter,
        semantic_acceptor=semantic_acceptor,
        candidate_limit=settings.candidate_cap,
    )
    return [
        OptimizationSnapshot(
            checkpoint=snapshot.checkpoint,
            representation="tensor_embeddings:random_mutation",
            attack_loss=snapshot.objective_score,
            counters=ComputeCounters(
                updates=snapshot.updates,
                forward_passes=(snapshot.updates + 1) * objective.forward_passes_per_evaluation,
                candidates_attempted=snapshot.candidates_attempted,
                candidates_accepted=snapshot.candidates_accepted,
                prompt_tokens=prompt_tokens,
            ),
            state={
                "z": snapshot.candidate.z.detach().to(device="cpu"),
                "u": snapshot.candidate.u.detach().to(device="cpu"),
                "z_token_ids": torch.empty((0, 0), dtype=torch.long),
                "u_token_ids": torch.empty((0, 0), dtype=torch.long),
            },
        )
        for snapshot in snapshots
    ]


def _run_local_qwen_tensor(
    loaded: LocalQwenHandle,
    record: BenchmarkExample,
    job: OptimizationJob,
    checkpoints: tuple[int, ...],
    settings: TensorOptimizationSettings,
) -> Iterable[OptimizationSnapshot]:
    method = tensor_method_for_recovery(job.method)
    if method not in _TENSOR_METHODS:
        raise ExecutionError(f"unsupported local tensor method: {job.method}")
    if method != job.method and job.method not in {
        f"{method}_recovery",
        f"{method}_recovery_eager",
        f"{method}_recovery_sdpa",
        f"{method}_recovery_eager_retry",
        f"{method}_recovery_checkpointed",
        f"{method}_recovery_rebalanced",
        f"{method}_recovery_fd",
        f"{method}_recovery_fd_sdpa",
    }:
        raise ExecutionError(f"unsupported local tensor recovery method: {job.method}")
    expected_checkpoints = (0,) if method == "init" else settings.checkpoints
    if checkpoints != expected_checkpoints:
        raise ExecutionError("local tensor executor requires the configured checkpoint policy")
    if loaded.tokenizer is None or loaded.model is None:
        raise ExecutionError("local Qwen handle is closed")

    encoded = loaded.tokenizer(record.attack_text, return_tensors="pt", add_special_tokens=True)
    token_ids = _input_ids(encoded)
    if not isinstance(token_ids, torch.Tensor):
        raise ExecutionError("local tensor executor tokenizer returned invalid token IDs")
    device = _embedding_device(loaded.model)
    token_ids = token_ids.to(device=device, dtype=torch.long)
    z_ids, u_ids = _initial_editable_token_ids(token_ids, settings)
    objective = TransformerAttackObjective(
        loaded.model,
        frozen_prompt_token_ids=token_ids,
        answer_token_ids=_anchor_token_ids(loaded.tokenizer, settings.answer_anchors, device),
        refusal_token_ids=_anchor_token_ids(loaded.tokenizer, settings.refusal_anchors, device),
        epsilon=settings.epsilon,
        lambda_fol=settings.lambda_fol,
        gamma_z=settings.gamma_z,
        gamma_u=settings.gamma_u,
    )
    embedding = _vocabulary_embeddings(loaded.model)
    prompt_tokens = _token_count(token_ids)
    if method == "random_mutation":
        return _random_mutation_snapshots(
            objective=objective,
            initial_state=objective.build_editable_state(z_ids, u_ids),
            settings=settings,
            seed=job.random_seed,
            prompt_tokens=prompt_tokens,
        )
    optimizer = (
        _baseline_optimizer(method, settings, embedding)
        if method in {"pez", "gbda", "gcg"}
        else _tensor_optimizer(method, settings)
    )
    snapshots = optimizer.run(
        objective,
        _initial_state_for_method(
            method,
            objective=objective,
            z_ids=z_ids,
            u_ids=u_ids,
            embedding=embedding,
        ),
        _tensor_ledger(method, settings),
        CheckpointEmitter(list(checkpoints)),
    )
    return [
        OptimizationSnapshot(
            checkpoint=snapshot.checkpoint,
            representation=(
                "init_token_embeddings"
                if method == "init"
                else f"tensor_embeddings:{getattr(snapshot, 'selection_branch', method)}"
            ),
            attack_loss=snapshot.attack_loss,
            fol=getattr(snapshot, "fol", None),
            internal_margin=getattr(snapshot, "internal_margin", None),
            counters=ComputeCounters(
                updates=snapshot.updates,
                forward_passes=snapshot.forward_passes,
                backward_passes=snapshot.backward_passes,
                hvp_calls=snapshot.hvp_calls,
                candidates_attempted=getattr(snapshot, "candidates_attempted", 0),
                candidates_accepted=getattr(snapshot, "candidates_accepted", 0),
                prompt_tokens=prompt_tokens,
            ),
            state=_checkpoint_state(snapshot),
        )
        for snapshot in snapshots
    ]


def build_local_qwen_tensor_executor(settings: TensorOptimizationSettings) -> SmokeExecutor:
    """Bind locked tensor settings to the local Qwen smoke executor."""

    def executor(
        loaded: object,
        record: BenchmarkExample,
        job: OptimizationJob,
        checkpoints: tuple[int, ...],
    ) -> Iterable[OptimizationSnapshot]:
        if not isinstance(loaded, LocalQwenHandle):
            raise ExecutionError("local tensor executor requires a local Qwen handle")
        return _run_local_qwen_tensor(loaded, record, job, checkpoints, settings)

    return executor


ModelLoader = Callable[[ResolvedModel], object]
SmokeExecutor = Callable[
    [object, BenchmarkExample, OptimizationJob, tuple[int, ...]],
    Iterable[OptimizationSnapshot],
]


@dataclass(frozen=True)
class _PreparedExecution:
    model: ResolvedModel
    run_id: str
    config_hash: str
    git_revision: str
    manifest_hash: str
    records: tuple[BenchmarkExample, ...]


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError(f"invalid {label}") from error
    if not isinstance(loaded, dict):
        raise ExecutionError(f"invalid {label}")
    return loaded


def _prepare_execution(request: ExecutionRequest) -> _PreparedExecution:
    if request.requested_limit < 1:
        raise ExecutionError("requested limit must be positive")
    if request.shard_count < 1:
        raise ExecutionError("shard count must be positive")
    if not 0 <= request.shard_index < request.shard_count:
        raise ExecutionError("shard index must be within the shard count")
    locked_config = request.output_root / request.locked_config_name
    if not locked_config.is_file():
        raise ExecutionError("locked config is missing")
    _load_json_object(locked_config, label="locked config")
    run_manifest = _load_json_object(request.output_root / "run_manifest.json", label="run manifest")
    config_hash = run_manifest.get("config_hash")
    run_id = run_manifest.get("run_id")
    git_revision = run_manifest.get("git_revision", "unavailable")
    if not all(isinstance(value, str) and value for value in (config_hash, run_id, git_revision)):
        raise ExecutionError("run manifest has incomplete identity")

    manifest_root = request.output_root / "manifests"
    header = ManifestHeader.model_validate(
        _load_json_object(manifest_root / f"controlled_{request.source}.header.json", label="manifest header")
    )
    if header.schema_version != request.schema_version or header.config_hash != config_hash:
        raise ExecutionError("locked manifest identity does not match the run")
    try:
        rows = tuple(BenchmarkExample.model_validate(row) for row in read_jsonl(manifest_root / f"controlled_{request.source}.jsonl"))
    except (TypeError, ValueError) as error:
        raise ExecutionError("manifest records are invalid") from error
    payloads = [row.model_dump(mode="json") for row in rows]
    if (
        len(rows) != header.record_count
        or tuple(row.example_id for row in rows) != header.ordered_example_ids
        or canonical_hash(payloads) != header.manifest_hash
    ):
        raise ExecutionError("locked manifest content does not match its header")
    if any(row.source != request.source for row in rows):
        raise ExecutionError("locked manifest source is invalid")
    if request.requested_limit > len(rows):
        raise ExecutionError("requested limit exceeds locked manifest size")
    try:
        model = validate_model_assets(request.local_model_path)
    except Exception as error:
        raise ExecutionError("offline local model validation failed") from error
    bounded = tuple(sorted(rows, key=lambda row: row.example_id)[: request.requested_limit])
    if request.requested_sample_ids is not None:
        requested_ids = tuple(request.requested_sample_ids)
        if not requested_ids or any(not sample_id for sample_id in requested_ids):
            raise ExecutionError("requested sample IDs must be non-empty")
        if len(requested_ids) != len(set(requested_ids)):
            raise ExecutionError("requested sample IDs must be unique")
        available = {row.example_id for row in bounded}
        if set(requested_ids) - available:
            raise ExecutionError("requested sample IDs are outside the bounded manifest")
        requested = set(requested_ids)
        bounded = tuple(row for row in bounded if row.example_id in requested)
    selected = tuple(
        row for index, row in enumerate(bounded) if index % request.shard_count == request.shard_index
    )
    return _PreparedExecution(
        model=model,
        run_id=run_id,
        config_hash=config_hash,
        git_revision=git_revision,
        manifest_hash=header.manifest_hash,
        records=selected,
    )


def _job_for(request: ExecutionRequest, prepared: _PreparedExecution, record: BenchmarkExample) -> OptimizationJob:
    cell_id = stable_id(
        "cell",
        {
            "config_hash": prepared.config_hash,
            "manifest_hash": prepared.manifest_hash,
            "method": request.method,
            "model_revision": prepared.model.revision,
            "source": request.source,
        },
    )
    seed = int(
        canonical_hash({"cell_id": cell_id, "sample_id": record.example_id, "seed": request.seed})[:16],
        16,
    ) % (2**31)
    return OptimizationJob(
        source=request.source,
        method=request.method,
        cell_id=cell_id,
        sample_id=record.example_id,
        random_seed=seed,
    )


def _write_failures(
    runner: OptimizationRunner,
    job: OptimizationJob,
    checkpoints: tuple[int, ...],
    *,
    failure_kind: FailureKind,
    failure_reason: str,
) -> int:
    path = runner.records_path(job)
    terminal = {
        row["checkpoint"]
        for row in read_jsonl(path)
        if (
            row.get("cell_id") == job.cell_id
            and row.get("sample_id") == job.sample_id
            and row.get("status") in {RecordStatus.complete.value, RecordStatus.failed.value}
            and isinstance(row.get("checkpoint"), int)
        )
    }
    ledger = JsonlLedger(path, key_fields=("cell_id", "sample_id", "checkpoint"))
    written = 0
    for checkpoint in checkpoints:
        if checkpoint in terminal:
            continue
        record = OptimizationRecord(
            schema_version=runner.schema_version,
            run_id=runner.run_id,
            config_hash=runner.config_hash,
            git_revision=runner.git_revision,
            cell_id=job.cell_id,
            sample_id=job.sample_id,
            source=job.source,
            method=job.method,
            checkpoint=checkpoint,
            random_seed=job.random_seed,
            status=RecordStatus.failed,
            failure_kind=failure_kind,
            failure_reason=failure_reason,
            state_path=None,
            representation="unavailable",
            attack_loss=None,
            fol=None,
            internal_margin=None,
            materialized_prompt=None,
            counters=ComputeCounters(),
        )
        if ledger.append_once(record.model_dump(mode="json")):
            written += 1
    return written


def run_execution(
    request: ExecutionRequest,
    *,
    mode: ExecutionMode,
    model_loader: ModelLoader | None = None,
    executor: SmokeExecutor | None = None,
) -> ExecutionSummary:
    """Preflight or run a bounded selection from one locked manifest."""
    prepared = _prepare_execution(request)
    if mode is ExecutionMode.dry_run:
        return ExecutionSummary(
            mode=mode,
            selected_records=len(prepared.records),
            completed_records=0,
            failed_records=0,
        )
    runner = OptimizationRunner(
        request.output_root,
        config_hash=prepared.config_hash,
        run_id=prepared.run_id,
        git_revision=prepared.git_revision,
        schema_version=request.schema_version,
    )
    if model_loader is None or executor is None:
        failed = sum(
            _write_failures(
                runner,
                _job_for(request, prepared, record),
                request.checkpoints,
                failure_kind=FailureKind.compatibility,
                failure_reason="local executor is unavailable",
            )
            for record in prepared.records
        )
        return ExecutionSummary(
            mode=mode,
            selected_records=len(prepared.records),
            completed_records=0,
            failed_records=failed,
        )

    try:
        model = model_loader(prepared.model)
    except Exception as error:
        failed = sum(
            _write_failures(
                runner,
                _job_for(request, prepared, record),
                request.checkpoints,
                failure_kind=FailureKind.compatibility,
                failure_reason=f"local model loader failed: {type(error).__name__}",
            )
            for record in prepared.records
        )
        return ExecutionSummary(
            mode=mode,
            selected_records=len(prepared.records),
            completed_records=0,
            failed_records=failed,
        )

    completed = 0
    failed = 0
    try:
        for record in prepared.records:
            job = _job_for(request, prepared, record)
            try:
                written = runner.run(
                    job,
                    checkpoints=request.checkpoints,
                    snapshot_factory=lambda checkpoints, record=record, job=job: executor(
                        model, record, job, checkpoints
                    ),
                )
                completed += len(written)
            except Exception as error:
                failed += _write_failures(
                    runner,
                    job,
                    request.checkpoints,
                    failure_kind=FailureKind.optimization,
                    failure_reason=f"executor failed: {type(error).__name__}",
                )
        return ExecutionSummary(
            mode=mode,
            selected_records=len(prepared.records),
            completed_records=completed,
            failed_records=failed,
        )
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()
