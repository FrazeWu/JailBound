from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

import benchmark.safety_eval.execution as execution
from benchmark.safety_eval.execution import (
    ExecutionError,
    ExecutionMode,
    ExecutionRequest,
    LocalQwenHandle,
    TensorOptimizationSettings,
    build_local_qwen_tensor_executor,
    load_local_qwen,
    local_qwen_init_executor,
    run_execution,
)


def test_tensor_recovery_sdpa_method_resolves_to_o_plus() -> None:
    from benchmark.safety_eval.execution import tensor_method_for_recovery

    assert tensor_method_for_recovery("jailbound_o_plus_recovery_sdpa") == "jailbound_o_plus"
    assert tensor_method_for_recovery("jailbound_o_plus_recovery_eager_retry") == "jailbound_o_plus"
    assert tensor_method_for_recovery("jailbound_o_plus_recovery_checkpointed") == "jailbound_o_plus"
    assert tensor_method_for_recovery("jailbound_o_plus_recovery_rebalanced") == "jailbound_o_plus"
    assert tensor_method_for_recovery("jailbound_o_plus_recovery_fd") == "jailbound_o_plus"
    assert tensor_method_for_recovery("jailbound_o_plus_recovery_fd_sdpa") == "jailbound_o_plus"
from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.manifest import write_controlled_manifest, write_v2_controlled_manifest
from benchmark.safety_eval.runner import OptimizationJob, OptimizationSnapshot
from benchmark.safety_eval.runtime import ResolvedModel
from benchmark.safety_eval.objective import EditableState
from benchmark.safety_eval.schema import (
    BenchmarkExample,
    ComputeCounters,
    FailureKind,
    RecordStatus,
    V2BenchmarkExample,
)


def _example(index: int) -> BenchmarkExample:
    text = f"neutral text {index}"
    return BenchmarkExample(
        example_id=f"fixture:{index:03d}",
        source="fixture",
        source_file="fixture.jsonl",
        source_row=index,
        source_sha256="a" * 64,
        intent=f"neutral intent {index}",
        attack_text=text,
        target_text=None,
        source_risk_label=None,
        source_attack_label="direct_request",
        risk_category="category-a",
        threat_domain="domain-a",
        attack_type="type-a",
        language="en",
        selection_stratum="category-a|type-a",
        selection_seed=20260725,
        prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        preprocessing=(),
    )


def _request(tmp_path: Path) -> ExecutionRequest:
    output_root = tmp_path / "output"
    config_hash = "b" * 64
    write_controlled_manifest(
        output_root,
        "fixture",
        (_example(2), _example(1)),
        source_file_sha256="a" * 64,
        config_hash=config_hash,
    )
    (output_root / "locked_config.json").write_text("{}", encoding="utf-8")
    (output_root / "run_manifest.json").write_text(
        json.dumps({"run_id": "run:fixture", "config_hash": config_hash, "git_revision": "fixture"}),
        encoding="utf-8",
    )
    model_path = tmp_path / "local-qwen"
    model_path.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model_path / name).write_text("{}", encoding="utf-8")
    return ExecutionRequest(
        output_root=output_root,
        locked_config_name="locked_config.json",
        schema_version="reviewer_eval.v1",
        local_model_path=model_path,
        source="fixture",
        method="fixture_method",
        checkpoints=(0, 25, 50, 100),
        requested_limit=1,
        seed=20260725,
    )


def _v2_request(tmp_path: Path) -> ExecutionRequest:
    request = _request(tmp_path)
    write_v2_controlled_manifest(
        request.output_root,
        "fixture",
        (_v2_example_with_middle_span(),),
        source_file_sha256="a" * 64,
        config_hash="b" * 64,
    )
    return replace(request, schema_version="reviewer_eval.v2")


def test_dry_run_validates_offline_assets_and_selects_locked_manifest_without_loading_model(tmp_path: Path) -> None:
    request = _request(tmp_path)

    summary = run_execution(
        request,
        mode=ExecutionMode.dry_run,
        model_loader=lambda _: (_ for _ in ()).throw(AssertionError("loader must not run")),
    )

    assert summary.mode is ExecutionMode.dry_run
    assert summary.selected_records == 1
    assert summary.completed_records == summary.failed_records == 0
    assert not (request.output_root / "optimization_config.json").exists()


def test_shards_partition_the_same_bounded_manifest_prefix_without_overlap(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), checkpoints=(0,), requested_limit=2, shard_count=2)
    executed: list[tuple[int, str]] = []

    def executor(_: object, record: BenchmarkExample, __: object, checkpoints: tuple[int, ...]):
        executed.append((record.source_row, record.example_id))
        return [
            OptimizationSnapshot(
                checkpoint=checkpoint,
                representation="fixture_ids",
                attack_loss=None,
                counters=ComputeCounters(),
            )
            for checkpoint in checkpoints
        ]

    for shard_index in range(2):
        run_execution(
            replace(request, shard_index=shard_index),
            mode=ExecutionMode.smoke,
            model_loader=lambda _: object(),
            executor=executor,
        )

    assert sorted(executed) == [(1, "fixture:001"), (2, "fixture:002")]
    assert len(executed) == len(set(executed))


def test_requested_sample_ids_select_only_the_registered_subset(tmp_path: Path) -> None:
    request = replace(
        _request(tmp_path),
        checkpoints=(0,),
        requested_limit=2,
        requested_sample_ids=("fixture:002",),
    )
    executed: list[str] = []

    def executor(_: object, record: BenchmarkExample, __: object, checkpoints: tuple[int, ...]):
        executed.append(record.example_id)
        return [
            OptimizationSnapshot(
                checkpoint=checkpoint,
                representation="fixture_ids",
                attack_loss=None,
                counters=ComputeCounters(),
            )
            for checkpoint in checkpoints
        ]

    summary = run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda _: object(),
        executor=executor,
    )

    assert summary.selected_records == 1
    assert executed == ["fixture:002"]


def test_smoke_uses_injected_loader_and_executor_to_write_checkpoint_ledger(tmp_path: Path) -> None:
    request = _request(tmp_path)
    loaded: list[object] = []
    executed: list[str] = []

    def executor(model: object, record: BenchmarkExample, _: object, checkpoints: tuple[int, ...]):
        assert model == "fake-local-model"
        executed.append(record.example_id)
        return [
            OptimizationSnapshot(
                checkpoint=checkpoint,
                representation="fixture_ids",
                attack_loss=float(checkpoint),
                counters=ComputeCounters(updates=checkpoint),
            )
            for checkpoint in checkpoints
        ]

    summary = run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda resolved: loaded.append(resolved) or "fake-local-model",
        executor=executor,
    )

    rows = read_jsonl(request.output_root / "optimization" / "fixture" / "fixture_method" / "records.jsonl")
    assert len(loaded) == 1
    assert executed == ["fixture:001"]
    assert summary.completed_records == 4
    assert summary.failed_records == 0
    assert {row["status"] for row in rows} == {RecordStatus.complete.value}
    assert {row["checkpoint"] for row in rows} == {0, 25, 50, 100}


def test_smoke_without_executor_writes_explicit_compatibility_failures(tmp_path: Path) -> None:
    request = _request(tmp_path)

    summary = run_execution(request, mode=ExecutionMode.smoke)

    rows = read_jsonl(request.output_root / "optimization" / "fixture" / "fixture_method" / "records.jsonl")
    assert summary.completed_records == 0
    assert summary.failed_records == 4
    assert {row["status"] for row in rows} == {RecordStatus.failed.value}
    assert {row["failure_kind"] for row in rows} == {FailureKind.compatibility.value}
    assert {row["checkpoint"] for row in rows} == {0, 25, 50, 100}


class _FakeIds:
    shape = (1, 3)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _: str, *, return_tensors: str, add_special_tokens: bool) -> dict[str, _FakeIds]:
        assert return_tensors == "pt"
        assert add_special_tokens is True
        self.calls += 1
        return {"input_ids": _FakeIds()}


class _FakeEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, token_ids: _FakeIds) -> object:
        assert isinstance(token_ids, _FakeIds)
        self.calls += 1
        return object()


class _FakeModel:
    def __init__(self) -> None:
        self.embedding = _FakeEmbedding()
        self.moved_to: list[str] = []

    def get_input_embeddings(self) -> _FakeEmbedding:
        return self.embedding

    def to(self, device: str) -> None:
        self.moved_to.append(device)


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))


def test_local_qwen_init_executor_uses_token_embeddings_at_checkpoint_zero_only(monkeypatch) -> None:
    _install_fake_torch(monkeypatch)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    handle = LocalQwenHandle(tokenizer=tokenizer, model=model)
    job = OptimizationJob(
        source="fixture",
        method="init",
        cell_id="cell:fixture",
        sample_id="fixture:001",
        random_seed=1,
    )

    snapshots = list(local_qwen_init_executor(handle, _example(1), job, (0,)))

    assert len(snapshots) == 1
    assert snapshots[0].checkpoint == 0
    assert snapshots[0].representation == "init_token_embeddings"
    assert snapshots[0].attack_loss is None
    assert snapshots[0].counters.prompt_tokens == 3
    assert tokenizer.calls == model.embedding.calls == 1
    with pytest.raises(ExecutionError, match="only method 'init'"):
        local_qwen_init_executor(handle, _example(1), job.__class__(**{**job.__dict__, "method": "zol"}), (0,))


def test_smoke_persists_concrete_local_qwen_init_snapshot_with_fake_model(monkeypatch, tmp_path: Path) -> None:
    _install_fake_torch(monkeypatch)
    request = replace(_request(tmp_path), method="init", checkpoints=(0,))
    model = _FakeModel()
    handle = LocalQwenHandle(tokenizer=_FakeTokenizer(), model=model)

    summary = run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda _: handle,
        executor=local_qwen_init_executor,
    )

    rows = read_jsonl(request.output_root / "optimization" / "fixture" / "init" / "records.jsonl")
    assert summary.completed_records == 1
    assert summary.failed_records == 0
    assert rows[0]["checkpoint"] == 0
    assert rows[0]["representation"] == "init_token_embeddings"
    assert rows[0]["attack_loss"] is None
    assert rows[0]["counters"]["prompt_tokens"] == 3
    assert handle.model is None
    assert model.moved_to == ["cpu"]


def test_local_qwen_loader_uses_offline_transformers_and_releases_model(monkeypatch, tmp_path: Path) -> None:
    tokenizer_calls: list[tuple[object, dict[str, object]]] = []
    model_calls: list[tuple[object, dict[str, object]]] = []
    model = _FakeModel()

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeTokenizer:
            tokenizer_calls.append((path, kwargs))
            return _FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeModel:
            model_calls.append((path, kwargs))
            return model

    model.eval = lambda: model  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizerFactory, AutoModelForCausalLM=FakeModelFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="fake-bfloat16", cuda=SimpleNamespace(is_available=lambda: False)),
    )

    handle = load_local_qwen(ResolvedModel(tmp_path, "local:fixture", "tokenizer", None))
    handle.close()

    assert tokenizer_calls == [(tmp_path, {"local_files_only": True})]
    assert model_calls == [
        (
            tmp_path,
            {
                "local_files_only": True,
                "torch_dtype": "fake-bfloat16",
                "device_map": "auto",
                "attn_implementation": "eager",
            },
        )
    ]
    assert model.moved_to == ["cpu"]


def test_local_qwen_loader_allows_an_explicit_sdpa_backend_for_recovery(monkeypatch, tmp_path: Path) -> None:
    model_calls: list[tuple[object, dict[str, object]]] = []
    model = _FakeModel()

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(_: object, **__: object) -> _FakeTokenizer:
            return _FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeModel:
            model_calls.append((path, kwargs))
            return model

    model.eval = lambda: model  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizerFactory, AutoModelForCausalLM=FakeModelFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="fake-bfloat16", cuda=SimpleNamespace(is_available=lambda: False)),
    )

    handle = load_local_qwen(
        ResolvedModel(tmp_path, "local:fixture", "tokenizer", None),
        attention_backend="sdpa",
    )
    handle.close()

    assert model_calls[0][1]["attn_implementation"] == "sdpa"


def test_local_qwen_loader_enables_nonreentrant_activation_checkpointing(monkeypatch, tmp_path: Path) -> None:
    model = _FakeModel()
    checkpoint_calls: list[dict[str, object]] = []
    train_calls: list[bool] = []
    model.eval = lambda: model  # type: ignore[attr-defined]
    model.train = lambda: train_calls.append(True) or model  # type: ignore[attr-defined]
    model.gradient_checkpointing_enable = lambda **kwargs: checkpoint_calls.append(kwargs)  # type: ignore[attr-defined]
    model.config = SimpleNamespace(attention_dropout=0.0)  # type: ignore[attr-defined]

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(_: object, **__: object) -> _FakeTokenizer:
            return _FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(_: object, **__: object) -> _FakeModel:
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizerFactory, AutoModelForCausalLM=FakeModelFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="fake-bfloat16", cuda=SimpleNamespace(is_available=lambda: False)),
    )

    handle = load_local_qwen(
        ResolvedModel(tmp_path, "local:fixture", "tokenizer", None),
        activation_checkpointing=True,
    )
    handle.close()

    assert checkpoint_calls == [{"gradient_checkpointing_kwargs": {"use_reentrant": False}}]
    assert train_calls == [True]


def test_local_qwen_loader_allows_balanced_device_placement(monkeypatch, tmp_path: Path) -> None:
    model_calls: list[tuple[object, dict[str, object]]] = []
    model = _FakeModel()
    model.eval = lambda: model  # type: ignore[attr-defined]

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(_: object, **__: object) -> _FakeTokenizer:
            return _FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeModel:
            model_calls.append((path, kwargs))
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizerFactory, AutoModelForCausalLM=FakeModelFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="fake-bfloat16", cuda=SimpleNamespace(is_available=lambda: False)),
    )

    handle = load_local_qwen(
        ResolvedModel(tmp_path, "local:fixture", "tokenizer", None),
        device_map="balanced",
    )
    handle.close()

    assert model_calls[0][1]["device_map"] == "balanced"


def test_local_qwen_loader_allows_an_explicit_device_map(monkeypatch, tmp_path: Path) -> None:
    model_calls: list[tuple[object, dict[str, object]]] = []
    model = _FakeModel()
    model.eval = lambda: model  # type: ignore[attr-defined]
    explicit_map = {"model.embed_tokens": 0, "model.layers.0": 0, "lm_head": 1}

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(_: object, **__: object) -> _FakeTokenizer:
            return _FakeTokenizer()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path: object, **kwargs: object) -> _FakeModel:
            model_calls.append((path, kwargs))
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizerFactory, AutoModelForCausalLM=FakeModelFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="fake-bfloat16", cuda=SimpleNamespace(is_available=lambda: False)),
    )

    handle = load_local_qwen(
        ResolvedModel(tmp_path, "local:fixture", "tokenizer", None),
        device_map=explicit_map,
    )
    handle.close()

    assert model_calls[0][1]["device_map"] == explicit_map


def test_two_gpu_recovery_map_splits_transformer_layers_evenly(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_fol_candidate_optimization.py"
    spec = importlib.util.spec_from_file_location("fol_candidate_optimization", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 28}), encoding="utf-8")

    mapping = module._two_gpu_recovery_device_map(tmp_path)

    assert sum(key.startswith("model.layers.") and value == 0 for key, value in mapping.items()) == 13
    assert sum(key.startswith("model.layers.") and value == 1 for key, value in mapping.items()) == 13
    assert mapping["lm_head"] == 1
    assert mapping["model.layers.13"] == "cpu"
    assert mapping["model.layers.27"] == "cpu"


def test_smoke_closes_loaded_local_qwen_handle_after_runner_writes_records(tmp_path: Path) -> None:
    request = _request(tmp_path)
    model = _FakeModel()
    handle = LocalQwenHandle(tokenizer=_FakeTokenizer(), model=model)

    def executor(_: object, _record: BenchmarkExample, _job: object, checkpoints: tuple[int, ...]):
        return [
            OptimizationSnapshot(
                checkpoint=checkpoint,
                representation="fixture_ids",
                attack_loss=None,
                counters=ComputeCounters(),
            )
            for checkpoint in checkpoints
        ]

    summary = run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda _: handle,
        executor=executor,
    )

    assert summary.completed_records == 4
    assert handle.model is None
    assert model.moved_to == ["cpu"]


class _TensorTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
        if kwargs == {"return_offsets_mapping": True}:
            assert text == "aa PAYLOAD zz"
            return {
                "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "offset_mapping": [(0, 2), (3, 6), (6, 10), (11, 13)],
            }
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": True}
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [1 if value == "answer" else 2]


class _OffsetTensorTokenizer(_TensorTokenizer):
    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
        if kwargs == {"return_offsets_mapping": True}:
            assert text == "aa PAYLOAD zz"
            return {
                "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
                "offset_mapping": [(0, 2), (3, 6), (6, 10), (11, 13)],
            }
        return super().__call__(text, **kwargs)  # type: ignore[arg-type]

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {"!": [7], "answer": [41, 42, 43], "refusal": [51, 52]}[value]


def _v2_example_with_middle_span() -> V2BenchmarkExample:
    example = _example(1)
    return V2BenchmarkExample.model_validate(
        {
            **example.model_dump(mode="json"),
            "schema_version": "reviewer_eval.v2",
            "intent_sha256": "b" * 64,
            "editable_spans": [
                {
                    "start": 3,
                    "end": 10,
                    "quote": "PAYLOAD",
                    "role": "harmful_payload",
                    "confidence": 1.0,
                    "rationale": "fixture",
                }
            ],
            "annotator_model": "fixture",
            "annotator_revision": "fixture-r1",
            "annotation_template_sha256": "c" * 64,
            "annotation_response_sha256": "d" * 64,
            "annotation_confidence": 1.0,
            "attack_text": "aa PAYLOAD zz",
            "prompt_sha256": hashlib.sha256(b"aa PAYLOAD zz").hexdigest(),
        }
    )


def test_tensor_executor_uses_annotated_middle_span_and_full_anchor_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class RecordingObjective:
        forward_passes_per_evaluation = 1

        def __init__(self, _model: object, **kwargs: object) -> None:
            observed.update(kwargs)

        def build_editable_state(self, z_ids: torch.Tensor, *_legacy: torch.Tensor) -> EditableState:
            observed["z_ids"] = z_ids.detach().clone()
            return EditableState(
                z=torch.zeros((1, z_ids.shape[1], 3), requires_grad=True),
                u=torch.zeros((1, 2, 3), requires_grad=True),
                z0=torch.zeros((1, z_ids.shape[1], 3)),
                u0=torch.zeros((1, 2, 3)),
            )

    class RecordingOptimizer:
        def run(self, _objective: object, state: EditableState, *_args: object) -> list[object]:
            observed["initial_state"] = state
            return []

    monkeypatch.setattr(execution, "TransformerAttackObjective", RecordingObjective)
    monkeypatch.setattr(execution, "_tensor_optimizer", lambda *_args: RecordingOptimizer())
    handle = LocalQwenHandle(tokenizer=_OffsetTensorTokenizer(), model=_TinyTensorCausalModel())
    job = OptimizationJob("fixture", "zol", "cell:fixture", "fixture:001", 1)

    assert list(execution._run_local_qwen_tensor(handle, _v2_example_with_middle_span(), job, (0, 1, 2), _tensor_settings())) == []
    prompt = observed["prompt"]
    assert prompt.editable_positions == (1, 2)
    assert tuple(ids.tolist() for ids in observed["answer_anchor_ids"]) == ([41, 42, 43],)
    assert tuple(ids.tolist() for ids in observed["refusal_anchor_ids"]) == ([51, 52],)


class _TinyTensorCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.embeddings = nn.Embedding(8, 3)
        self.projection = nn.Linear(3, 8, bias=False)
        self.moved_to: list[str] = []
        self.forward_calls = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool):
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert use_cache is False
        self.forward_calls += 1
        return SimpleNamespace(logits=self.projection(inputs_embeds.cumsum(dim=1)))

    def to(self, device: str):
        self.moved_to.append(device)
        return self


def _tensor_settings(*, checkpoints: tuple[int, ...] = (0, 1, 2)) -> TensorOptimizationSettings:
    return TensorOptimizationSettings(
        checkpoints=checkpoints,
        update_budget=2,
        dual_branch_updates={"o_minus": 1, "o_plus": 1},
        candidate_cap=3,
        prefix_tokens=2,
        learning_rate=0.01,
        lambda_fol=0.1,
        epsilon=0.1,
        gamma_z=0.01,
        gamma_u=0.01,
        grad_clip=1.0,
        answer_anchors=("answer",),
        refusal_anchors=("refusal",),
    )


def test_tensor_settings_enable_finite_difference_fol_only_when_requested() -> None:
    from benchmark.safety_eval.execution import _tensor_optimizer

    optimizer = _tensor_optimizer(
        "jailbound_o_plus",
        replace(_tensor_settings(), finite_difference_fol=True),
    )

    assert optimizer.finite_difference_fol is True


def test_tensor_smoke_zol_maps_optimizer_snapshots_to_runner_records_and_closes_model(tmp_path: Path) -> None:
    request = replace(_v2_request(tmp_path), method="zol", checkpoints=(0, 1, 2))
    model = _TinyTensorCausalModel()
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=model)

    summary = run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda _: handle,
        executor=build_local_qwen_tensor_executor(_tensor_settings()),
    )

    rows = read_jsonl(request.output_root / "optimization" / "fixture" / "zol" / "records.jsonl")
    assert summary.completed_records == 3
    assert summary.failed_records == 0
    assert [row["checkpoint"] for row in rows] == [0, 1, 2]
    assert [row["counters"]["updates"] for row in rows] == [0, 1, 2]
    assert all(row["representation"] == "tensor_embeddings:zol" for row in rows)
    assert all(row["attack_loss"] is not None and row["internal_margin"] is not None for row in rows)
    assert all(row["fol"] is None for row in rows)
    assert rows[-1]["counters"]["forward_passes"] == model.forward_calls
    assert handle.model is None
    assert model.moved_to == ["cpu"]
    assert all(parameter.grad is None for parameter in model.parameters())


def test_tensor_smoke_persists_content_free_checkpoint_state(tmp_path: Path) -> None:
    request = replace(_v2_request(tmp_path), method="pez", checkpoints=(0, 1, 2))
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=_TinyTensorCausalModel())

    run_execution(
        request,
        mode=ExecutionMode.smoke,
        model_loader=lambda _: handle,
        executor=build_local_qwen_tensor_executor(_tensor_settings()),
    )

    rows = read_jsonl(request.output_root / "optimization" / "fixture" / "pez" / "records.jsonl")
    paths = [Path(row["state_path"]) for row in rows]
    assert all(path.is_file() for path in paths)
    payload = torch.load(paths[-1], weights_only=True)
    assert set(payload) == {
        "base_token_ids",
        "editable_positions",
        "editable_span_hashes",
        "tokenizer_revision",
        "u",
        "u_token_ids",
        "z",
        "z_token_ids",
    }
    assert payload["base_token_ids"].tolist() == [[1, 2, 3, 4]]
    assert payload["editable_positions"].tolist() == [1, 2]
    assert payload["tokenizer_revision"] == "local-tokenizer"
    assert len(payload["editable_span_hashes"]) == 1
    assert payload["z"].ndim == 3
    assert payload["u"].ndim == 3
    assert payload["z_token_ids"].ndim == payload["u_token_ids"].ndim == 2


def test_tensor_smoke_supports_budgeted_random_mutation_without_text_output() -> None:
    settings = _tensor_settings(checkpoints=(0, 25, 50, 100))
    settings = replace(settings, update_budget=100, candidate_cap=100)
    executor = build_local_qwen_tensor_executor(settings)
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=_TinyTensorCausalModel())

    snapshots = list(
        executor(handle, _v2_example_with_middle_span(), OptimizationJob("fixture", "random_mutation", "cell:fixture", "fixture:001", 9), (0, 25, 50, 100))
    )

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 25, 50, 100]
    assert [snapshot.counters.updates for snapshot in snapshots] == [0, 25, 50, 100]
    assert [snapshot.counters.forward_passes for snapshot in snapshots] == [1, 26, 51, 101]
    assert snapshots[-1].counters.candidates_attempted == 100
    assert all(snapshot.representation == "tensor_embeddings:random_mutation" for snapshot in snapshots)


@pytest.mark.parametrize("method", ["pez", "gbda", "gcg"])
def test_tensor_smoke_dispatches_discrete_and_continuous_baselines(method: str) -> None:
    executor = build_local_qwen_tensor_executor(_tensor_settings())
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=_TinyTensorCausalModel())

    snapshots = list(executor(handle, _v2_example_with_middle_span(), OptimizationJob("fixture", method, "cell:fixture", "fixture:001", 1), (0, 1, 2)))

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 2]
    assert {snapshot.representation for snapshot in snapshots} == {f"tensor_embeddings:{method}"}
    assert all(snapshot.attack_loss is not None for snapshot in snapshots)
    assert all(snapshot.fol is None for snapshot in snapshots)


def test_tensor_smoke_dispatches_dual_branch_and_rejects_unknown_method() -> None:
    model = _TinyTensorCausalModel()
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=model)
    executor = build_local_qwen_tensor_executor(_tensor_settings())
    dual_job = OptimizationJob("fixture", "dual_branch", "cell:fixture", "fixture:001", 1)

    snapshots = list(executor(handle, _v2_example_with_middle_span(), dual_job, (0, 1, 2)))

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 2]
    assert all(snapshot.fol is not None for snapshot in snapshots)
    assert {snapshot.representation for snapshot in snapshots} <= {
        "tensor_embeddings:o_minus",
        "tensor_embeddings:o_plus",
    }
    with pytest.raises(ExecutionError, match="unsupported local tensor method"):
        list(executor(handle, _example(1), OptimizationJob("fixture", "unknown", "cell:fixture", "fixture:001", 1), (0, 1, 2)))


def test_tensor_smoke_init_uses_transformer_objective_at_checkpoint_zero_only() -> None:
    executor = build_local_qwen_tensor_executor(_tensor_settings())
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=_TinyTensorCausalModel())
    job = OptimizationJob("fixture", "init", "cell:fixture", "fixture:001", 1)

    snapshots = list(executor(handle, _v2_example_with_middle_span(), job, (0,)))

    assert len(snapshots) == 1
    assert snapshots[0].checkpoint == 0
    assert snapshots[0].representation == "init_token_embeddings"
    assert snapshots[0].attack_loss is not None
    with pytest.raises(ExecutionError, match="configured checkpoint policy"):
        list(executor(handle, _example(1), job, (0, 1, 2)))


@pytest.mark.parametrize("method", ["jailbound_o_minus", "jailbound_o_plus"])
def test_tensor_smoke_supports_each_fol_branch_with_fake_model(method: str) -> None:
    executor = build_local_qwen_tensor_executor(_tensor_settings())
    handle = LocalQwenHandle(tokenizer=_TensorTokenizer(), model=_TinyTensorCausalModel())
    job = OptimizationJob("fixture", method, "cell:fixture", "fixture:001", 1)

    snapshots = list(executor(handle, _v2_example_with_middle_span(), job, (0, 1, 2)))

    assert [snapshot.checkpoint for snapshot in snapshots] == [0, 1, 2]
    assert all(snapshot.fol is not None for snapshot in snapshots)
    assert {snapshot.representation for snapshot in snapshots} == {f"tensor_embeddings:{method}"}
