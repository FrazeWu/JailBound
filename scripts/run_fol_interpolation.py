"""Generate fixed, semantically screened FOL interpolation paths without logging text."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.anchor_scorer import tokenize_anchor_set
from benchmark.safety_eval.execution import load_local_qwen
from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.fol_boundary import (
    LabeledEditableState,
    interpolate_joint_states,
    random_joint_direction,
    select_nearest_opposite_label_pairs,
)
from benchmark.safety_eval.io import atomic_write_jsonl, canonical_hash, read_jsonl
from benchmark.safety_eval.materialization import materialize_checkpoint
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.prompt_contract import tokenize_editable_prompt
from benchmark.safety_eval.pipeline import generate_materialized_records, write_stage_records
from benchmark.safety_eval.runtime import validate_model_assets
from benchmark.safety_eval.schema import JudgmentRecord, OptimizationRecord, RecordStatus, V2BenchmarkExample, stable_id
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder
from benchmark.safety_eval.transformer_objective import TransformerAttackObjective


ROOT = Path(__file__).resolve().parents[1]


def _semantic_threshold(root: Path) -> float:
    payload = json.loads((root.parent / "manifests" / "semantic_calibration.json").read_text(encoding="utf-8"))
    value = payload.get("threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError("semantic calibration threshold is invalid")
    return float(value)


def _run_identity(root: Path) -> tuple[str, str]:
    payload = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    run_id, config_hash = payload.get("run_id"), payload.get("config_hash")
    if not isinstance(run_id, str) or not isinstance(config_hash, str):
        raise ValueError("FOL run manifest is invalid")
    return run_id, config_hash


def _records(root: Path, source: str) -> tuple[dict[str, V2BenchmarkExample], dict[str, OptimizationRecord]]:
    examples = {
        row.example_id: row
        for row in (V2BenchmarkExample.model_validate(row) for row in read_jsonl(root / "manifests" / "v2" / f"controlled_{source}.jsonl"))
    }
    optimized = {
        row.sample_id: row
        for row in (OptimizationRecord.model_validate(row) for row in resolved_terminal_payloads(root, source).values())
        if row.checkpoint == 100 and row.status is RecordStatus.complete and row.state_path
    }
    if set(examples) != set(optimized):
        raise ValueError(f"FOL terminal O+ states are incomplete for {source}")
    return examples, optimized


def _primary_labels(root: Path, *, source: str, target_key: str, judge_key: str, threshold: float) -> dict[str, bool]:
    path = root / "judgments" / judge_key / target_key / source / "fol_boundary" / "records.jsonl"
    labels: dict[str, bool] = {}
    for payload in read_jsonl(path):
        row = JudgmentRecord.model_validate(payload)
        if row.checkpoint != 0 or row.threshold != threshold:
            continue
        if row.status is not RecordStatus.complete:
            continue
        if row.sample_id in labels:
            raise ValueError("duplicate primary FOL label")
        labels[row.sample_id] = row.unsafe_label
    return labels


def _candidate_states(
    root: Path,
    *,
    source: str,
    target_key: str,
    judge_key: str,
    threshold: float,
) -> tuple[dict[str, dict[str, object]], tuple[LabeledEditableState, ...]]:
    examples, optimized = _records(root, source)
    labels = _primary_labels(root, source=source, target_key=target_key, judge_key=judge_key, threshold=threshold)
    metadata = [row for row in read_jsonl(root / "selected_perturbations.jsonl") if row.get("source") == source]
    states: dict[str, dict[str, object]] = {}
    selection: list[LabeledEditableState] = []
    for row in metadata:
        candidate_id, sample_id, kind = row.get("perturbation_id"), row.get("sample_id"), row.get("kind")
        if not all(isinstance(value, str) and value for value in (candidate_id, sample_id, kind)):
            raise ValueError("FOL perturbation metadata is invalid")
        if sample_id not in examples or sample_id not in optimized:
            raise ValueError("FOL perturbation references an unknown terminal state")
        if candidate_id not in labels:
            continue
        payload = torch.load(str(optimized[sample_id].state_path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("FOL terminal state is invalid")
        z, u = payload.get("z"), payload.get("u")
        if not isinstance(z, torch.Tensor) or not isinstance(u, torch.Tensor):
            raise ValueError("FOL terminal state has no editable tensors")
        z, u = z.detach().float(), u.detach().float()
        if kind == "perturbation":
            radius, direction_seed = row.get("radius"), row.get("direction_seed")
            if (
                isinstance(radius, bool)
                or not isinstance(radius, (int, float))
                or isinstance(direction_seed, bool)
                or not isinstance(direction_seed, int)
            ):
                raise ValueError("FOL perturbation metadata is invalid")
            generator = torch.Generator(device="cpu").manual_seed(direction_seed)
            dz, du = random_joint_direction(tuple(z.shape), tuple(u.shape), generator)
            z, u = z + float(radius) * dz, u + float(radius) * du
        elif kind != "baseline":
            raise ValueError("unknown FOL perturbation kind")
        states[candidate_id] = {
            "source": source,
            "sample_id": sample_id,
            "example": examples[sample_id],
            "base": optimized[sample_id],
            "z": z,
            "u": u,
        }
        selection.append(LabeledEditableState(
            sample_id=sample_id,
            candidate_id=candidate_id,
            unsafe_label=labels[candidate_id],
            vector=torch.cat((z.reshape(-1), u.reshape(-1))).cpu(),
        ))
    return states, tuple(selection)


def _objective(
    *,
    model: Any,
    tokenizer: Any,
    example: V2BenchmarkExample,
    config: Any,
) -> TransformerAttackObjective:
    prompt = tokenize_editable_prompt(example.attack_text, example.editable_spans, tokenizer, "fol-interpolation")
    return TransformerAttackObjective(
        model,
        prompt=prompt,
        answer_anchor_ids=tokenize_anchor_set(tokenizer, tuple(config.optimization.answer_anchors)),
        refusal_anchor_ids=tokenize_anchor_set(tokenizer, tuple(config.optimization.refusal_anchors)),
        epsilon=config.optimization.epsilon,
        lambda_fol=config.optimization.lambda_fol,
        gamma_z=config.optimization.gamma_z,
        gamma_u=config.optimization.gamma_u,
    )


def _state_metrics(
    objective: TransformerAttackObjective,
    *,
    z: torch.Tensor,
    u: torch.Tensor,
    path_id: str,
    point_index: int,
    hvp_directions: int,
) -> tuple[float, float]:
    def leaf() -> EditableState:
        return EditableState(
            z=z.detach().clone().requires_grad_(True),
            u=u.detach().clone().requires_grad_(True),
            z0=z.detach().clone(),
            u0=u.detach().clone(),
        )

    value = objective.evaluate(leaf(), include_fol=True)
    if value.fol is None:
        raise ValueError("FOL interpolation objective did not return FOL")
    fol = float(value.fol.detach().cpu())
    curvature_values: list[float] = []
    for direction_index in range(hvp_directions):
        seed = int(canonical_hash({"path": path_id, "point": point_index, "direction": direction_index})[:16], 16) % (2**31)
        state = leaf()
        dz, du = random_joint_direction(tuple(state.z.shape), tuple(state.u.shape), torch.Generator(device="cpu").manual_seed(seed))
        dz, du = dz.to(device=state.z.device, dtype=state.z.dtype), du.to(device=state.u.device, dtype=state.u.dtype)
        hz, hu = objective.hvp(state, (dz, du))
        curvature_values.append(abs(float((hz * dz).sum().detach().cpu() + (hu * du).sum().detach().cpu())))
    return fol, sum(curvature_values) / len(curvature_values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = ROOT / config.run.output_root / "fol_boundary"
    model_path = config.models.surrogate.local_path
    if model_path is None:
        raise ValueError("FOL interpolation requires the local model")
    run_id, config_hash = _run_identity(root)
    sources = tuple(config.fol.sources)
    all_states: dict[str, dict[str, object]] = {}
    candidates: list[LabeledEditableState] = []
    for source in sources:
        states, source_candidates = _candidate_states(
            root,
            source=source,
            target_key=config.models.targets[0].key,
            judge_key=config.judging.primary.key,
            threshold=config.judging.primary.threshold,
        )
        all_states.update(states)
        candidates.extend(source_candidates)
    pairs = select_nearest_opposite_label_pairs(candidates)
    threshold = _semantic_threshold(root)
    resolved = validate_model_assets(model_path)
    handle = load_local_qwen(resolved)
    try:
        if handle.model is None or handle.tokenizer is None:
            raise ValueError("local FOL interpolation model did not load")
        embedding = handle.model.get_input_embeddings().weight.detach()
        encoder = QwenHiddenMeanEncoder(model_path, tokenizer=handle.tokenizer, model=handle.model, revision=resolved.revision)
        materializations = []
        point_rows: list[dict[str, object]] = []
        for pair in pairs:
            safe = all_states[pair.safe_candidate_id]
            unsafe = all_states[pair.unsafe_candidate_id]
            if safe["source"] != unsafe["source"] or safe["sample_id"] != unsafe["sample_id"]:
                raise ValueError("interpolation pair spans multiple prompts")
            source, sample_id = str(safe["source"]), str(safe["sample_id"])
            example = safe["example"]
            if not isinstance(example, BenchmarkExample):
                raise ValueError("interpolation example is invalid")
            path_id = stable_id("fol-interpolation-path", {
                "source": source,
                "sample_id": sample_id,
                "safe": pair.safe_candidate_id,
                "unsafe": pair.unsafe_candidate_id,
            })
            safe_z, safe_u = safe["z"], safe["u"]
            unsafe_z, unsafe_u = unsafe["z"], unsafe["u"]
            if not all(isinstance(value, torch.Tensor) for value in (safe_z, safe_u, unsafe_z, unsafe_u)):
                raise ValueError("interpolation endpoint state is invalid")
            objective = _objective(model=handle.model, tokenizer=handle.tokenizer, example=example, config=config)
            device, dtype = embedding.device, embedding.dtype
            states = interpolate_joint_states(
                safe_z.to(device=device, dtype=dtype),
                safe_u.to(device=device, dtype=dtype),
                unsafe_z.to(device=device, dtype=dtype),
                unsafe_u.to(device=device, dtype=dtype),
            )
            for point_index, (z, u) in enumerate(states):
                interpolation_id = stable_id("fol-interpolation-point", {"path_id": path_id, "point_index": point_index})
                provisional = materialize_checkpoint(
                    state_payload={"z": z, "u": u}, vocabulary_embeddings=embedding, tokenizer=handle.tokenizer,
                    schema_version=config.run.schema_version, run_id=run_id, config_hash=config_hash,
                    sample_id=interpolation_id, source=source, method="fol_interpolation", checkpoint=point_index,
                    original_prompt=example.attack_text, category=example.risk_category,
                    semantic_similarity=1.0, semantic_threshold=0.0,
                )
                semantic_vectors = encoder.encode([example.attack_text, provisional.flat_prompt])
                semantic_similarity = float(semantic_vectors[0] @ semantic_vectors[1])
                record = materialize_checkpoint(
                    state_payload={"z": z, "u": u}, vocabulary_embeddings=embedding, tokenizer=handle.tokenizer,
                    schema_version=config.run.schema_version, run_id=run_id, config_hash=config_hash,
                    sample_id=interpolation_id, source=source, method="fol_interpolation", checkpoint=point_index,
                    original_prompt=example.attack_text, category=example.risk_category,
                    semantic_similarity=semantic_similarity, semantic_threshold=threshold,
                )
                fol, curvature = _state_metrics(
                    objective, z=z, u=u, path_id=path_id, point_index=point_index,
                    hvp_directions=config.fol.hvp_directions,
                )
                materializations.append(record.model_dump(mode="json"))
                point_rows.append({
                    "path_id": path_id,
                    "interpolation_id": interpolation_id,
                    "source": source,
                    "sample_id": sample_id,
                    "point_index": point_index,
                    "safe_endpoint_id": pair.safe_candidate_id,
                    "unsafe_endpoint_id": pair.unsafe_candidate_id,
                    "endpoint_distance": pair.distance,
                    "semantic_accepted": record.status is RecordStatus.complete,
                    "fol": fol,
                    "curvature": curvature,
                })
        atomic_write_jsonl(root / "interpolation_points.jsonl", sorted(point_rows, key=lambda row: (str(row["path_id"]), int(row["point_index"]))))
        materialized = write_stage_records(root, stage="interpolation_materialization", rows=materializations)
        generated = generate_materialized_records(
            root, materializations, model=handle.model, tokenizer=handle.tokenizer,
            target_key=config.models.targets[0].key, target_revision=resolved.revision,
            max_new_tokens=config.judging.max_new_tokens,
        )
    finally:
        handle.close()
    print(json.dumps({
        "generated": generated.written_records,
        "materialized": materialized.written_records,
        "paths": len(pairs),
        "points": len(point_rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
