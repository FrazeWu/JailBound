from __future__ import annotations

from pathlib import Path

import torch

from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.pipeline import (
    judge_response_records,
    generate_materialized_records,
    materialize_records_from_disk,
    materialize_optimization_record,
    StageSummary,
    failed_optimization_materialization,
    select_final_optimization_records,
    write_materialization_records,
    write_stage_records,
)
from benchmark.safety_eval.schema import BenchmarkExample, FailureKind, OptimizationRecord, RecordStatus


def test_write_stage_records_resumes_per_source_method_checkpoint_without_duplicates(tmp_path: Path) -> None:
    rows = (
        {
            "sample_id": "source_a:001",
            "source": "source_a",
            "method": "method_a",
            "checkpoint": 0,
            "status": "complete",
        },
        {
            "sample_id": "source_a:001",
            "source": "source_a",
            "method": "method_a",
            "checkpoint": 25,
            "status": "failed",
        },
        {
            "sample_id": "source_b:001",
            "source": "source_b",
            "method": "method_b",
            "checkpoint": 0,
            "status": "complete",
        },
    )

    first = write_stage_records(tmp_path, stage="materializations", rows=rows)
    second = write_stage_records(tmp_path, stage="materializations", rows=rows)

    assert first == StageSummary(selected_records=3, written_records=3, failed_records=1)
    assert second == StageSummary(selected_records=3, written_records=0, failed_records=1)
    assert len(read_jsonl(tmp_path / "materializations" / "source_a" / "method_a" / "records.jsonl")) == 2
    assert len(read_jsonl(tmp_path / "materializations" / "source_b" / "method_b" / "records.jsonl")) == 1


def test_write_stage_records_rejects_unsafe_path_components(tmp_path: Path) -> None:
    rows = (
        {
            "sample_id": "sample:001",
            "source": "../source",
            "method": "method_a",
            "checkpoint": 0,
            "status": "complete",
        },
    )

    try:
        write_stage_records(tmp_path, stage="materializations", rows=rows)
    except ValueError as error:
        assert "path component" in str(error)
    else:
        raise AssertionError("unsafe stage path must be rejected")


def test_write_stage_records_partitions_judgments_and_keeps_threshold_rows_distinct(tmp_path: Path) -> None:
    rows = (
        {
            "sample_id": "source_a:001",
            "source": "source_a",
            "method": "method_a",
            "checkpoint": 100,
            "target_key": "target_a",
            "judge_key": "judge_a",
            "threshold": 0.4,
            "status": "complete",
        },
        {
            "sample_id": "source_a:001",
            "source": "source_a",
            "method": "method_a",
            "checkpoint": 100,
            "target_key": "target_a",
            "judge_key": "judge_a",
            "threshold": 0.6,
            "status": "complete",
        },
    )

    summary = write_stage_records(
        tmp_path,
        stage="judgments",
        rows=rows,
        partition_fields=("target_key", "judge_key"),
        key_fields=("sample_id", "checkpoint", "threshold"),
    )

    assert summary == StageSummary(selected_records=2, written_records=2, failed_records=0)
    assert len(
        read_jsonl(tmp_path / "judgments" / "target_a" / "judge_a" / "source_a" / "method_a" / "records.jsonl")
    ) == 2


def test_failed_optimization_is_preserved_as_a_failed_materialization() -> None:
    optimization = OptimizationRecord.model_validate(
        {
            "schema_version": "reviewer_eval.v1",
            "run_id": "run:fixture",
            "config_hash": "a" * 64,
            "git_revision": "fixture",
            "cell_id": "cell:fixture",
            "sample_id": "source_a:001",
            "source": "source_a",
            "method": "method_a",
            "checkpoint": 25,
            "random_seed": 7,
            "status": "failed",
            "failure_kind": "optimization",
            "failure_reason": "optimizer failure",
            "state_path": None,
            "representation": "fixture",
            "attack_loss": None,
            "fol": None,
            "internal_margin": None,
            "materialized_prompt": None,
            "counters": {},
        }
    )

    materialization = failed_optimization_materialization(optimization, category="category_a")

    assert materialization.status is RecordStatus.failed
    assert materialization.failure_kind is FailureKind.optimization
    assert materialization.sample_id == optimization.sample_id
    assert materialization.checkpoint == 25
    assert materialization.intent_preserved is False


def test_final_candidate_selection_uses_only_init_zero_and_optimizer_hundred() -> None:
    def record(method: str, checkpoint: int) -> OptimizationRecord:
        return OptimizationRecord.model_validate(
            {
                "schema_version": "reviewer_eval.v1",
                "run_id": "run:fixture",
                "config_hash": "a" * 64,
                "git_revision": "fixture",
                "cell_id": f"cell:{method}",
                "sample_id": "source_a:001",
                "source": "source_a",
                "method": method,
                "checkpoint": checkpoint,
                "random_seed": 7,
                "status": "complete",
                "failure_kind": None,
                "failure_reason": None,
                "state_path": "state.pt",
                "representation": "fixture",
                "attack_loss": None,
                "fol": None,
                "internal_margin": None,
                "materialized_prompt": None,
                "counters": {},
            }
        )

    selected = select_final_optimization_records(
        (record("init", 0), record("zol", 0), record("zol", 25), record("zol", 50), record("zol", 100))
    )

    assert [(row.method, row.checkpoint) for row in selected] == [("init", 0), ("zol", 100)]


def test_write_materialization_records_uses_the_locked_optimization_layout(tmp_path: Path) -> None:
    row = {
        "sample_id": "source_a:001",
        "source": "source_a",
        "method": "method_a",
        "checkpoint": 100,
        "status": "complete",
    }

    summary = write_materialization_records(tmp_path, (row,))

    assert summary == StageSummary(selected_records=1, written_records=1, failed_records=0)
    assert read_jsonl(tmp_path / "optimization" / "source_a" / "method_a" / "materialization.jsonl") == [row]


def test_materialize_optimization_record_loads_the_saved_state_and_preserves_checkpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "state.pt"
    torch.save(
        {
            "z": torch.zeros((1, 1, 2)),
            "u": torch.zeros((1, 1, 2)),
            "z_token_ids": torch.tensor([[1]]),
            "u_token_ids": torch.tensor([[2]]),
        },
        state_path,
    )
    optimization = OptimizationRecord.model_validate(
        {
            "schema_version": "reviewer_eval.v1",
            "run_id": "run:fixture",
            "config_hash": "a" * 64,
            "git_revision": "fixture",
            "cell_id": "cell:fixture",
            "sample_id": "source_a:001",
            "source": "source_a",
            "method": "method_a",
            "checkpoint": 100,
            "random_seed": 7,
            "status": "complete",
            "failure_kind": None,
            "failure_reason": None,
            "state_path": str(state_path),
            "representation": "fixture",
            "attack_loss": None,
            "fol": None,
            "internal_margin": None,
            "materialized_prompt": None,
            "counters": {},
        }
    )
    example = BenchmarkExample.model_validate(
        {
            "example_id": "source_a:001",
            "source": "source_a",
            "source_file": "fixture.jsonl",
            "source_row": 1,
            "source_sha256": "b" * 64,
            "intent": "fixture intent",
            "attack_text": "fixture request",
            "target_text": None,
            "source_risk_label": None,
            "source_attack_label": "direct_request",
            "risk_category": "category_a",
            "threat_domain": "domain_a",
            "attack_type": "direct_request",
            "language": "en",
            "selection_stratum": "category_a|direct_request",
            "selection_seed": 7,
            "prompt_sha256": "c" * 64,
            "preprocessing": (),
        }
    )

    class Tokenizer:
        all_special_ids: list[int] = []

        @staticmethod
        def decode(ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return {1: "prefix", 2: "suffix"}[ids[0]]

    record = materialize_optimization_record(
        optimization,
        example=example,
        vocabulary_embeddings=torch.eye(3, 2),
        tokenizer=Tokenizer(),
        semantic_similarity=lambda before, after: 1.0,
        semantic_threshold=0.9,
    )

    assert record.status is RecordStatus.complete
    assert record.checkpoint == 100
    assert record.prefix_token_ids == (1,)
    assert record.seed_token_ids == (2,)


def test_materialize_records_from_disk_joins_only_the_immutable_matching_manifest(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    example = {
        "example_id": "source_a:001",
        "source": "source_a",
        "source_file": "fixture.jsonl",
        "source_row": 1,
        "source_sha256": "b" * 64,
        "intent": "fixture intent",
        "attack_text": "fixture request",
        "target_text": None,
        "source_risk_label": None,
        "source_attack_label": "direct_request",
        "risk_category": "category_a",
        "threat_domain": "domain_a",
        "attack_type": "direct_request",
        "language": "en",
        "selection_stratum": "category_a|direct_request",
        "selection_seed": 7,
        "prompt_sha256": "c" * 64,
        "preprocessing": [],
    }
    (manifest_root / "controlled_source_a.jsonl").write_text(__import__("json").dumps(example) + "\n", encoding="utf-8")
    state_path = tmp_path / "state.pt"
    torch.save({"z": torch.zeros((1, 1, 2)), "u": torch.zeros((1, 1, 2)), "z_token_ids": torch.tensor([[1]]), "u_token_ids": torch.tensor([[2]])}, state_path)
    optimization_root = tmp_path / "optimization" / "source_a" / "method_a"
    optimization_root.mkdir(parents=True)
    optimization = {
        "schema_version": "reviewer_eval.v1", "run_id": "run:fixture", "config_hash": "a" * 64,
        "git_revision": "fixture", "cell_id": "cell:fixture", "sample_id": "source_a:001",
        "source": "source_a", "method": "method_a", "checkpoint": 100, "random_seed": 7,
        "status": "complete", "failure_kind": None, "failure_reason": None, "state_path": str(state_path),
        "representation": "fixture", "attack_loss": None, "fol": None, "internal_margin": None,
        "materialized_prompt": None, "counters": {},
    }
    (optimization_root / "records.jsonl").write_text(__import__("json").dumps(optimization) + "\n", encoding="utf-8")

    class Tokenizer:
        all_special_ids: list[int] = []

        @staticmethod
        def decode(ids: list[int], *, skip_special_tokens: bool) -> str:
            return {1: "prefix", 2: "suffix"}[ids[0]]

    summary = materialize_records_from_disk(
        tmp_path,
        vocabulary_embeddings=torch.eye(3, 2),
        tokenizer=Tokenizer(),
        semantic_similarity=lambda before, after: 1.0,
        semantic_threshold=0.9,
        final_only=True,
    )

    assert summary == StageSummary(selected_records=1, written_records=1, failed_records=0)
    assert len(read_jsonl(optimization_root / "materialization.jsonl")) == 1


def test_generate_materialized_records_partitions_by_target(tmp_path: Path) -> None:
    materialization = {
        "schema_version": "reviewer_eval.v1", "run_id": "run:fixture", "config_hash": "a" * 64,
        "sample_id": "source_a:001", "source": "source_a", "method": "init", "checkpoint": 0,
        "system_prompt": "", "user_prompt": "fixture", "flat_prompt": "fixture", "prefix_token_ids": [1], "seed_token_ids": [2],
        "prefix_projection_cosine": 1.0, "seed_projection_cosine": 1.0, "semantic_similarity_before": 1.0, "semantic_similarity_after": 1.0,
        "category_before": "a", "category_after": "a", "intent_preserved": True, "projection_attack_score_before": None, "projection_attack_score_after": None,
        "status": "complete", "failure_kind": None, "failure_reason": None,
    }

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs): return [[1]]
        def decode(self, *args, **kwargs): return "ok"
    class Model:
        def generate(self, ids, **kwargs): return [[1, 2]]

    summary = generate_materialized_records(tmp_path, (materialization,), model=Model(), tokenizer=Tokenizer(), target_key="qwen", target_revision="local", max_new_tokens=4)
    assert summary == StageSummary(selected_records=1, written_records=1, failed_records=0)
    assert len(read_jsonl(tmp_path / "responses" / "qwen" / "source_a" / "init" / "records.jsonl")) == 1


def test_generate_materialized_records_skips_already_terminal_response_keys(tmp_path: Path) -> None:
    materialization = {
        "schema_version": "reviewer_eval.v1", "run_id": "run:fixture", "config_hash": "a" * 64,
        "sample_id": "source_a:001", "source": "source_a", "method": "init", "checkpoint": 0,
        "system_prompt": "", "user_prompt": "fixture", "flat_prompt": "fixture", "prefix_token_ids": [1], "seed_token_ids": [2],
        "prefix_projection_cosine": 1.0, "seed_projection_cosine": 1.0, "semantic_similarity_before": 1.0, "semantic_similarity_after": 1.0,
        "category_before": "a", "category_after": "a", "intent_preserved": True, "projection_attack_score_before": None, "projection_attack_score_after": None,
        "status": "complete", "failure_kind": None, "failure_reason": None,
    }

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs): return [[1]]
        def decode(self, *args, **kwargs): return "ok"

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, ids, **kwargs):
            self.calls += 1
            return [[1, 2]]

    model = Model()
    first = generate_materialized_records(tmp_path, (materialization,), model=model, tokenizer=Tokenizer(), target_key="qwen", target_revision="local", max_new_tokens=4)
    second = generate_materialized_records(tmp_path, (materialization,), model=model, tokenizer=Tokenizer(), target_key="qwen", target_revision="local", max_new_tokens=4)

    assert first.written_records == 1
    assert second == StageSummary(selected_records=1, written_records=0, failed_records=0)
    assert model.calls == 1


def test_judge_response_records_partitions_by_judge_and_threshold(tmp_path: Path) -> None:
    response = {"schema_version":"reviewer_eval.v1","run_id":"run:fixture","config_hash":"a"*64,"sample_id":"s:1","source":"s","method":"init","checkpoint":0,"target_key":"qwen","target_revision":"local","prompt_hash":"b"*64,"response":"ok","input_tokens":1,"generated_tokens":1,"status":"complete","failure_kind":None,"failure_reason":None}
    class Judge:
        key = "judge"
        revision = "local"
        def judge_response(self, **kwargs):
            return {key: kwargs[key] for key in ("run_id", "config_hash", "sample_id", "source", "method", "checkpoint", "target_key", "threshold")} | {"schema_version":"reviewer_eval.v1", "judge_key":self.key, "judge_revision":self.revision, "unsafe_score":0.1, "unsafe_label":False, "refusal_label":True, "raw_output":"", "status":"complete", "failure_kind":None, "failure_reason":None}
    summary = judge_response_records(tmp_path, (response,), judge=Judge(), threshold=0.5)
    assert summary == StageSummary(1, 1, 0)
    assert len(read_jsonl(tmp_path / "judgments" / "judge" / "qwen" / "s" / "init" / "records.jsonl")) == 1


def test_judge_response_records_skips_already_terminal_judgment_keys(tmp_path: Path) -> None:
    response = {"schema_version":"reviewer_eval.v1","run_id":"run:fixture","config_hash":"a"*64,"sample_id":"s:1","source":"s","method":"init","checkpoint":0,"target_key":"qwen","target_revision":"local","prompt_hash":"b"*64,"response":"ok","input_tokens":1,"generated_tokens":1,"status":"complete","failure_kind":None,"failure_reason":None}

    class Judge:
        key = "judge"
        revision = "local"

        def __init__(self) -> None:
            self.calls = 0

        def judge_response(self, **kwargs):
            self.calls += 1
            return {key: kwargs[key] for key in ("run_id", "config_hash", "sample_id", "source", "method", "checkpoint", "target_key", "threshold")} | {"schema_version":"reviewer_eval.v1", "judge_key":self.key, "judge_revision":self.revision, "unsafe_score":0.1, "unsafe_label":False, "refusal_label":True, "raw_output":"", "status":"complete", "failure_kind":None, "failure_reason":None}

    judge = Judge()
    first = judge_response_records(tmp_path, (response,), judge=judge, threshold=0.5)
    second = judge_response_records(tmp_path, (response,), judge=judge, threshold=0.5)

    assert first.written_records == 1
    assert second == StageSummary(1, 0, 0)
    assert judge.calls == 1
