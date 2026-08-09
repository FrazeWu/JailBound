from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.runtime import (
    PreflightError,
    git_provenance,
    lock_runtime_config,
    validate_model_assets,
)
from benchmark.safety_eval.io import canonical_hash, sha256_file
from benchmark.safety_eval.schema import RecordStatus, TransportType, V2MaterializationRecord
from benchmark.safety_eval.runner import OptimizationJob, stable_state_id


ROOT = Path(__file__).resolve().parents[2]


def _write_model_snapshot(path: Path, names: tuple[str, ...]) -> None:
    path.mkdir()
    for index, name in enumerate(names):
        asset = path / name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text(f"asset-{index}:{name}\n", encoding="utf-8")


def test_validate_model_assets_accepts_complete_local_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_text("{}", encoding="utf-8")
    resolved = validate_model_assets(model)
    assert resolved.revision.startswith("local-sha256:")
    assert resolved.tokenizer_hash


@pytest.mark.parametrize(
    "changed_asset",
    (
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
        "nested/loader_metadata.json",
    ),
)
def test_validate_model_assets_revision_covers_every_regular_snapshot_file(
    tmp_path: Path, changed_asset: str
) -> None:
    model = tmp_path / "model"
    names = (
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
        "nested/loader_metadata.json",
    )
    _write_model_snapshot(model, names)
    original = validate_model_assets(model)

    with (model / changed_asset).open("a", encoding="utf-8") as stream:
        stream.write("changed\n")

    changed = validate_model_assets(model)
    assert changed.revision != original.revision


@pytest.mark.parametrize(
    "changed_asset",
    (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ),
)
def test_validate_model_assets_tokenizer_hash_covers_tokenizer_loader_inputs(
    tmp_path: Path, changed_asset: str
) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(
        model,
        (
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "model.safetensors",
        ),
    )
    original = validate_model_assets(model)

    with (model / changed_asset).open("a", encoding="utf-8") as stream:
        stream.write("changed\n")

    assert validate_model_assets(model).tokenizer_hash != original.tokenizer_hash


def test_validate_model_assets_identity_is_independent_of_root_and_creation_order(
    tmp_path: Path,
) -> None:
    names = (
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "model.safetensors",
        "nested/metadata.json",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_model_snapshot(first, names)
    _write_model_snapshot(second, tuple(reversed(names)))
    for name in names:
        (second / name).write_bytes((first / name).read_bytes())

    first_resolved = validate_model_assets(first)
    second_resolved = validate_model_assets(second)

    assert first_resolved.revision == second_resolved.revision
    assert first_resolved.tokenizer_hash == second_resolved.tokenizer_hash


def test_validate_model_assets_identity_ignores_downloader_metadata(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(
        model, ("config.json", "tokenizer.json", "model.safetensors")
    )
    metadata = model / ".cache/huggingface/download/config.json.lock"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("download-state-a\n", encoding="utf-8")
    original = validate_model_assets(model)

    metadata.write_text("download-state-b\n", encoding="utf-8")

    assert validate_model_assets(model) == original


def test_validate_model_assets_hashes_symlinked_loader_inputs(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    _write_model_snapshot(
        blobs, ("config.json", "tokenizer.json", "model.safetensors")
    )
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).symlink_to(blobs / name)

    original = validate_model_assets(model)
    with (blobs / "model.safetensors").open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    changed = validate_model_assets(model)

    assert changed.revision != original.revision
    assert changed.tokenizer_hash == original.tokenizer_hash


def test_validate_model_assets_rejects_incomplete_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PreflightError, match="incomplete model snapshot"):
        validate_model_assets(model)


def test_validate_model_assets_accepts_vocab_as_tokenizer_asset(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(model, ("config.json", "vocab.json", "model.safetensors"))

    assert validate_model_assets(model).tokenizer_hash


@pytest.mark.parametrize("missing_asset", ("config.json", "vocab.json", "model.safetensors"))
def test_validate_model_assets_requires_each_snapshot_asset_class(
    tmp_path: Path, missing_asset: str
) -> None:
    model = tmp_path / "model"
    names = {"config.json", "vocab.json", "model.safetensors"} - {missing_asset}
    _write_model_snapshot(model, tuple(sorted(names)))

    with pytest.raises(PreflightError, match="incomplete model snapshot"):
        validate_model_assets(model)


def test_validate_model_assets_tokenizer_hash_ignores_weight_changes(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    _write_model_snapshot(
        model, ("config.json", "tokenizer.json", "model.safetensors")
    )
    original = validate_model_assets(model)

    with (model / "model.safetensors").open("a", encoding="utf-8") as stream:
        stream.write("changed\n")
    changed = validate_model_assets(model)

    assert changed.revision != original.revision
    assert changed.tokenizer_hash == original.tokenizer_hash


def test_lock_runtime_config_writes_content_addressed_identity(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/benchmark/safety_eval_additions.yaml")
    locked = lock_runtime_config(config, output_root=tmp_path, source_hashes={"advbench": "a" * 64})
    assert locked.run_id.startswith("run:")
    assert (tmp_path / "locked_config.json").exists()
    assert json.loads((tmp_path / "run_manifest.json").read_text())["run_id"] == locked.run_id


def test_git_provenance_is_independent_of_process_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(ROOT)
    inside = git_provenance(ROOT)
    monkeypatch.chdir(elsewhere)
    outside = git_provenance(ROOT)

    assert outside == inside
    assert set(outside) == {"git_revision", "git_status_hash"}


def test_lock_runtime_config_is_byte_identical_across_process_cwds(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config(ROOT / "configs/benchmark/safety_eval_additions.yaml")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    monkeypatch.chdir(ROOT)
    lock_runtime_config(
        config,
        output_root=first_root,
        source_hashes={"advbench": "a" * 64},
        repository_root=ROOT,
    )
    monkeypatch.chdir(elsewhere)
    lock_runtime_config(
        config,
        output_root=second_root,
        source_hashes={"advbench": "a" * 64},
        repository_root=ROOT,
    )

    for name in ("locked_config.json", "run_manifest.json"):
        assert (first_root / name).read_bytes() == (second_root / name).read_bytes()


def test_v2_runtime_rejects_legacy_manifest_artifacts(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_output_root

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "controlled_fixture.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="schema-v1 artifact"):
        validate_v2_output_root(tmp_path)


def test_v2_runtime_rejects_legacy_response_provenance(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    response = tmp_path / "responses" / "target" / "source" / "method" / "records.jsonl"
    response.parent.mkdir(parents=True)
    response.write_text('{"schema_version":"reviewer_eval.v2"}\n', encoding="utf-8")

    with pytest.raises(PreflightError, match="legacy or invalid v2 response"):
        validate_v2_provenance_ledgers(tmp_path)


def test_v2_materialization_membership_rejects_a_same_hash_tampered_ledger_payload(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import require_v2_materialization_ledger_membership

    payload = {
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "a" * 64,
        "sample_id": "s:1", "source": "s", "method": "method", "branch": "method", "step": 1,
        "transport": TransportType.text, "state_sha256": "b" * 64, "surrogate_tokenizer_sha256": "c" * 64, "surrogate_embedding_sha256": "d" * 64,
        "editable_positions": (1,), "original_token_ids": (3, 4), "projected_z_token_ids": (7,),
        "projected_u_token_ids": (8,), "reconstructed_base_token_ids": (3, 8), "complete_token_ids": (7, 3, 8),
        "frozen_positions_unchanged": True, "span_boundary_expansions": ((0, 1),),
        "full_prompt_similarity": 0.5, "editable_span_similarity": 0.0, "flat_prompt": "fixture",
        "status": RecordStatus.complete, "failure_kind": None, "failure_reason": None,
    }
    record = V2MaterializationRecord.model_validate({**payload, "materialization_sha256": canonical_hash(payload)})
    ledger = tmp_path / "optimization" / "s" / "method" / "materialization.jsonl"
    ledger.parent.mkdir(parents=True)
    tampered = record.model_dump(mode="json") | {"flat_prompt": "tampered"}
    ledger.write_text(json.dumps(tampered) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="payload does not match"):
        require_v2_materialization_ledger_membership(tmp_path, record)


def _write_v2_provenance_fixture(
    tmp_path: Path, *, base_dtype: torch.dtype = torch.long
) -> tuple[Path, V2MaterializationRecord]:
    """Write one internally consistent optimization/materialization pair."""
    from benchmark.safety_eval.materialization import vocabulary_embedding_sha256

    job = OptimizationJob("s", "method", "cell:fixture", "s:1", 1)
    state_path = (
        tmp_path / "optimization" / "s" / "method" / "states"
        / f"{stable_state_id(job, 1)}.pt"
    )
    state_path.parent.mkdir(parents=True)
    span = {
        "start": 0, "end": 1, "quote": "f", "role": "harmful_payload",
        "confidence": 1.0, "rationale": "fixture",
    }
    span_hash = canonical_hash(span)
    torch.save(
        {
            "z": torch.tensor([[[1.0, 0.0]]]),
            "u": torch.tensor([[[0.0, 1.0]]]),
            "base_token_ids": torch.tensor([[3, 4]], dtype=base_dtype),
            "editable_positions": torch.tensor([1]),
            "tokenizer_revision": "c" * 64,
            "editable_span_hashes": (span_hash,),
            "input_embedding_sha256": vocabulary_embedding_sha256(torch.eye(3, 2)),
        },
        state_path,
    )
    materialization_payload = {
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "a" * 64,
        "sample_id": "s:1", "source": "s", "method": "method", "branch": "method", "step": 1,
        "transport": TransportType.text, "state_sha256": sha256_file(state_path),
        "surrogate_tokenizer_sha256": "c" * 64,
        "surrogate_embedding_sha256": vocabulary_embedding_sha256(torch.eye(3, 2)),
        "editable_positions": (1,), "original_token_ids": (3, 4), "projected_z_token_ids": (7,),
        "projected_u_token_ids": (8,), "reconstructed_base_token_ids": (3, 8), "complete_token_ids": (7, 3, 8),
        "frozen_positions_unchanged": True, "span_boundary_expansions": ((0, 1),),
        "full_prompt_similarity": 0.5, "editable_span_similarity": 0.0, "flat_prompt": "fixture",
        "status": RecordStatus.complete, "failure_kind": None, "failure_reason": None,
    }
    materialization = V2MaterializationRecord.model_validate(
        {**materialization_payload, "materialization_sha256": canonical_hash(materialization_payload)}
    )
    directory = tmp_path / "optimization" / "s" / "method"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifests" / "v2" / "controlled_s.jsonl"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "schema_version": "reviewer_eval.v2", "example_id": "s:1", "source": "s",
        "source_file": "fixture", "source_row": 1, "source_sha256": "a" * 64,
        "intent": "fixture", "attack_text": "fixture", "target_text": None,
        "source_risk_label": None, "source_attack_label": "direct_request",
        "risk_category": "risk", "threat_domain": "domain", "attack_type": "direct_request",
        "language": "en", "selection_stratum": "risk|direct_request", "selection_seed": 1,
        "prompt_sha256": "b" * 64, "preprocessing": [], "intent_sha256": "c" * 64,
        "editable_spans": [span], "annotator_model": "model", "annotator_revision": "revision",
        "annotation_template_sha256": "d" * 64, "annotation_response_sha256": "e" * 64,
        "annotation_confidence": 1.0,
    }) + "\n", encoding="utf-8")
    optimization = {
        "schema_version": "reviewer_eval.v2", "run_id": "run:fixture", "config_hash": "a" * 64,
        "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "s:1", "source": "s",
        "method": "method", "checkpoint": 1, "random_seed": 1, "status": "complete",
        "failure_kind": None, "failure_reason": None, "state_path": str(state_path),
        "state_sha256": sha256_file(state_path), "representation": "tensor_embeddings:method",
        "attack_loss": None, "fol": None, "internal_margin": None, "materialized_prompt": None, "counters": {},
    }
    (directory / "records.jsonl").write_text(json.dumps(optimization) + "\n", encoding="utf-8")
    (directory / "materialization.jsonl").write_text(
        json.dumps(materialization.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    return state_path, materialization


def test_v2_provenance_rejects_materialization_without_matching_optimization(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    _, materialization = _write_v2_provenance_fixture(tmp_path)
    record_path = tmp_path / "optimization" / "s" / "method" / "records.jsonl"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["checkpoint"] = materialization.step + 1
    record_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="matching optimization"):
        validate_v2_provenance_ledgers(tmp_path)


def test_v2_provenance_rejects_missing_materialized_state_before_resume(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    state_path, _ = _write_v2_provenance_fixture(tmp_path)
    state_path.unlink()

    with pytest.raises(PreflightError, match="state file is missing"):
        validate_v2_provenance_ledgers(tmp_path)


def test_v2_provenance_rejects_noninteger_base_tokens_even_with_matching_digests(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    _write_v2_provenance_fixture(tmp_path, base_dtype=torch.float32)

    with pytest.raises(PreflightError, match="base-token contract"):
        validate_v2_provenance_ledgers(tmp_path)


def test_v2_output_root_rejects_a_legacy_record_after_a_valid_first_row(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_output_root

    _write_v2_provenance_fixture(tmp_path)
    records = tmp_path / "optimization" / "s" / "method" / "records.jsonl"
    with records.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version":"reviewer_eval.v1"}\n')

    with pytest.raises(PreflightError, match="schema-v1 artifact"):
        validate_v2_output_root(tmp_path)


def test_v2_provenance_revalidates_editable_span_hashes_against_manifest(tmp_path: Path) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    state_path, _ = _write_v2_provenance_fixture(tmp_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["editable_span_hashes"] = ("f" * 64,)
    torch.save(state, state_path)
    state_sha256 = sha256_file(state_path)

    records_path = tmp_path / "optimization" / "s" / "method" / "records.jsonl"
    optimization = json.loads(records_path.read_text(encoding="utf-8"))
    optimization["state_sha256"] = state_sha256
    records_path.write_text(json.dumps(optimization) + "\n", encoding="utf-8")

    materialization_path = tmp_path / "optimization" / "s" / "method" / "materialization.jsonl"
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    materialization["state_sha256"] = state_sha256
    materialization.pop("materialization_sha256")
    materialization["materialization_sha256"] = canonical_hash(materialization)
    materialization_path.write_text(json.dumps(materialization) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="editable-span contract"):
        validate_v2_provenance_ledgers(tmp_path)


@pytest.mark.parametrize("state_mode", ("external", "symlink"))
def test_v2_provenance_requires_the_runner_managed_nonsymlink_state_path(
    tmp_path: Path, state_mode: str
) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    state_path, _ = _write_v2_provenance_fixture(tmp_path)
    if state_mode == "external":
        external = tmp_path / "external.pt"
        external.write_bytes(state_path.read_bytes())
        records_path = tmp_path / "optimization" / "s" / "method" / "records.jsonl"
        record = json.loads(records_path.read_text(encoding="utf-8"))
        record["state_path"] = str(external)
        records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        expected = "runner-managed location"
    else:
        external = tmp_path / "external.pt"
        external.write_bytes(state_path.read_bytes())
        state_path.unlink()
        state_path.symlink_to(external)
        expected = "cannot be a symlink"

    with pytest.raises(PreflightError, match=expected):
        validate_v2_provenance_ledgers(tmp_path)


@pytest.mark.parametrize("symlink_ancestor", ("optimization", "source", "method", "states"))
def test_v2_provenance_rejects_symlinked_runner_state_ancestors(
    tmp_path: Path, symlink_ancestor: str
) -> None:
    """Every managed checkpoint ancestor must be a real directory."""
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    state_path, _ = _write_v2_provenance_fixture(tmp_path)
    root = tmp_path
    components = {
        "optimization": root / "optimization",
        "source": root / "optimization" / "s",
        "method": root / "optimization" / "s" / "method",
        "states": state_path.parent,
    }
    ancestor = components[symlink_ancestor]
    relocated = tmp_path / f"relocated-{symlink_ancestor}"
    ancestor.rename(relocated)
    ancestor.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(PreflightError, match="cannot be a symlink"):
        validate_v2_provenance_ledgers(tmp_path)


def _write_v2_response_provenance_fixture(tmp_path: Path) -> Path:
    """Persist one response whose complete provenance joins the projection."""
    _, materialization = _write_v2_provenance_fixture(tmp_path)
    from benchmark.safety_eval.schema import V2ResponseRecord, token_ids_sha256

    executed_token_ids_sha256 = token_ids_sha256(materialization.complete_token_ids)
    response = V2ResponseRecord(
        schema_version="reviewer_eval.v2",
        run_id=materialization.run_id,
        config_hash=materialization.config_hash,
        sample_id=materialization.sample_id,
        source=materialization.source,
        method=materialization.method,
        checkpoint=materialization.step,
        target_key="target",
        target_revision="local-sha256:" + "f" * 64,
        prompt_hash=executed_token_ids_sha256,
        response="fixture response",
        input_tokens=len(materialization.complete_token_ids),
        generated_tokens=1,
        status=RecordStatus.complete,
        failure_kind=None,
        failure_reason=None,
        branch=materialization.branch,
        state_step=materialization.step,
        transport=materialization.transport,
        materialization_sha256=materialization.materialization_sha256,
        target_tokenizer_sha256=materialization.surrogate_tokenizer_sha256,
        executed_token_ids_sha256=executed_token_ids_sha256,
    )
    path = tmp_path / "responses" / "target" / "s" / "method" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(response.model_dump(mode="json")) + "\n", encoding="utf-8")
    return path


def test_v2_provenance_rejects_duplicate_target_execution_response(tmp_path: Path) -> None:
    """A judgment must never be reached through an ambiguous target execution."""
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    response_path = _write_v2_response_provenance_fixture(tmp_path)
    response_path.write_text(
        response_path.read_text(encoding="utf-8") * 2,
        encoding="utf-8",
    )

    with pytest.raises(PreflightError, match="duplicate v2 target execution response"):
        validate_v2_provenance_ledgers(tmp_path)


def test_v2_provenance_rejects_duplicate_judgment_identity(tmp_path: Path) -> None:
    """A resumed v2 judgment ledger cannot silently retain duplicate rows."""
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    response_path = _write_v2_response_provenance_fixture(tmp_path)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    judgment = {
        "schema_version": "reviewer_eval.v2",
        "run_id": response["run_id"], "config_hash": response["config_hash"],
        "sample_id": response["sample_id"], "source": response["source"],
        "method": response["method"], "checkpoint": response["checkpoint"],
        "target_key": response["target_key"], "judge_key": "judge",
        "judge_revision": "judge:fixture", "unsafe_score": 0.0,
        "unsafe_label": False, "refusal_label": None, "threshold": 0.5,
        "raw_output": "", "status": "complete", "failure_kind": None,
        "failure_reason": None, "branch": response["branch"],
        "state_step": response["state_step"], "transport": response["transport"],
        "materialization_sha256": response["materialization_sha256"],
        "target_revision": response["target_revision"],
        "target_tokenizer_sha256": response["target_tokenizer_sha256"],
    }
    path = tmp_path / "judgments" / "judge" / "target" / "s" / "method" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(judgment) + "\n" + json.dumps(judgment) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="duplicate v2 judgment identity"):
        validate_v2_provenance_ledgers(tmp_path)


def test_v2_provenance_rejects_a_judgment_for_a_stale_target_snapshot(tmp_path: Path) -> None:
    """A judgment cannot be resumed after the executed target identity changes."""
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    response_path = _write_v2_response_provenance_fixture(tmp_path)
    response = json.loads(response_path.read_text(encoding="utf-8"))
    judgment = {
        "schema_version": "reviewer_eval.v2",
        "run_id": response["run_id"], "config_hash": response["config_hash"],
        "sample_id": response["sample_id"], "source": response["source"],
        "method": response["method"], "checkpoint": response["checkpoint"],
        "target_key": response["target_key"], "judge_key": "judge",
        "judge_revision": "judge:fixture", "unsafe_score": 0.0,
        "unsafe_label": False, "refusal_label": None, "threshold": 0.5,
        "raw_output": "", "status": "complete", "failure_kind": None,
        "failure_reason": None, "branch": response["branch"],
        "state_step": response["state_step"], "transport": response["transport"],
        "materialization_sha256": response["materialization_sha256"],
        "target_revision": "local-sha256:" + "d" * 64,
        "target_tokenizer_sha256": response["target_tokenizer_sha256"],
    }
    path = tmp_path / "judgments" / "judge" / "target" / "s" / "method" / "records.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(judgment) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="does not join exactly one response"):
        validate_v2_provenance_ledgers(tmp_path)


@pytest.mark.parametrize(
    "field,tampered",
    (
        ("run_id", "run:other"),
        ("config_hash", "b" * 64),
        ("sample_id", "s:other"),
        ("source", "other-source"),
        ("method", "other-method"),
        ("checkpoint", 2),
        ("state_step", 2),
        ("branch", "other-branch"),
    ),
)
def test_v2_provenance_rejects_response_identity_not_matching_materialization(
    tmp_path: Path, field: str, tampered: str | int
) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    path = _write_v2_response_provenance_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = tampered
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match="response identity does not match materialization"):
        validate_v2_provenance_ledgers(tmp_path)


@pytest.mark.parametrize(
    "partition,expected",
    (("wrong-target", "target"), ("other-source", "source"), ("other-method", "method")),
)
def test_v2_provenance_rejects_response_ledger_in_wrong_partition(
    tmp_path: Path, partition: str, expected: str
) -> None:
    from benchmark.safety_eval.runtime import validate_v2_provenance_ledgers

    path = _write_v2_response_provenance_fixture(tmp_path)
    parts = ["target", "s", "method"]
    parts[{"target": 0, "source": 1, "method": 2}[expected]] = partition
    misplaced = tmp_path / "responses" / parts[0] / parts[1] / parts[2] / "records.jsonl"
    misplaced.parent.mkdir(parents=True)
    path.rename(misplaced)

    with pytest.raises(PreflightError, match=f"response ledger {expected} partition"):
        validate_v2_provenance_ledgers(tmp_path)
