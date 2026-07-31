from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_materialization_ablation.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("materialization_ablation_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example(sample_id: str, source: str) -> dict[str, object]:
    return {
        "example_id": sample_id,
        "source": source,
        "source_file": "fixture.jsonl",
        "source_row": 0,
        "source_sha256": "a" * 64,
        "intent": "reference intent",
        "attack_text": "seed request",
        "target_text": None,
        "source_risk_label": None,
        "source_attack_label": "fixture",
        "risk_category": "fixture",
        "threat_domain": "fixture",
        "attack_type": "fixture",
        "language": "en",
        "selection_stratum": "fixture",
        "selection_seed": 7,
        "prompt_sha256": "b" * 64,
        "preprocessing": [],
    }


def _optimization(
    sample_id: str,
    source: str,
    method: str,
    checkpoint: int,
    state_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": "reviewer_eval.v1",
        "run_id": "run:fixture",
        "config_hash": "c" * 64,
        "git_revision": "d" * 40,
        "cell_id": f"cell:{source}:{method}",
        "sample_id": sample_id,
        "source": source,
        "method": method,
        "checkpoint": checkpoint,
        "random_seed": 7,
        "status": "complete",
        "failure_kind": None,
        "failure_reason": None,
        "state_path": str(state_path),
        "representation": f"tensor_embeddings:{method}",
        "attack_loss": 0.0,
        "fol": 0.0,
        "internal_margin": 0.0,
        "materialized_prompt": None,
        "counters": {
            "updates": checkpoint,
            "forward_passes": 1,
            "backward_passes": 1,
            "hvp_calls": 0,
            "candidates_attempted": 0,
            "candidates_accepted": 0,
            "prompt_tokens": 3,
            "generated_tokens": 0,
            "judged_tokens": 0,
            "wall_seconds": 0.0,
            "peak_gpu_bytes": 0,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _study(tmp_path: Path, *, hidden_size: int = 2) -> Path:
    root = tmp_path / "matched"
    output = tmp_path / "evidence"
    source = "harmbench"
    sample_id = "harmbench:fixture"
    methods = ("jailbound_o_minus", "jailbound_o_plus")
    _write_jsonl(root / "manifests" / f"controlled_{source}.jsonl", [_example(sample_id, source)])
    all_rows: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    for method in methods:
        for checkpoint in (0, 100):
            state_path = root / "optimization" / source / method / "states" / f"{checkpoint}.pt"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "z": torch.ones((1, 1, hidden_size)),
                    "u": torch.ones((1, 1, hidden_size)),
                },
                state_path,
            )
            all_rows[method].append(_optimization(sample_id, source, method, checkpoint, state_path))
        _write_jsonl(root / "optimization" / source / method / "records.jsonl", all_rows[method])
    (root / "locked_config.json").write_text(
        json.dumps({
            "run": {"schema_version": "reviewer_eval.v1"},
            "models": {
                "surrogate": {
                    "key": "qwen2_5_7b",
                    "repo_id": "Qwen/Qwen2.5-7B-Instruct",
                    "local_path": str(tmp_path / "model"),
                    "revision": None,
                }
            },
            "optimization": {"update_budget": 100},
            "judging": {"max_new_tokens": 512},
        }),
        encoding="utf-8",
    )
    config = tmp_path / "ablation.yaml"
    config.write_text(
        f"""source_root: {root}\noutput_root: {output}\nsources: [harmbench]\nexpected_samples_per_source: 1\ncheckpoint: 100\nbranch_methods:\n  High-Value: jailbound_o_minus\n  Safety-Sensitivity: jailbound_o_plus\nmodel:\n  key: qwen2_5_7b\n  repo_id: Qwen/Qwen2.5-7B-Instruct\n  local_path: {tmp_path / 'model'}\n  revision: local\n  hidden_size: {hidden_size}\n  attention_implementation: eager\nmax_new_tokens: 512\n""",
        encoding="utf-8",
    )
    return config


def test_load_locked_inputs_resolves_both_branches_and_both_checkpoints(tmp_path: Path) -> None:
    module = _load_script()

    locked = module.load_locked_inputs(_study(tmp_path))

    assert locked.branch_methods == {
        module.Branch.high_value: "jailbound_o_minus",
        module.Branch.safety_sensitivity: "jailbound_o_plus",
    }
    assert locked.checkpoint == 100
    assert locked.sources == ("harmbench",)
    assert len(locked.units) == 2
    assert all(unit.error is None for unit in locked.units)
    assert all(unit.initial_state_path.is_file() and unit.final_state_path.is_file() for unit in locked.units)
    assert all(len(unit.final_state_sha256) == 64 for unit in locked.units)


def test_load_locked_inputs_rejects_wrong_qwen_model_identity(tmp_path: Path) -> None:
    module = _load_script()
    config = _study(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("key: qwen2_5_7b", "key: qwen2_5_14b"), encoding="utf-8")

    with pytest.raises(ValueError, match="Qwen2.5-7B"):
        module.load_locked_inputs(config)


@pytest.mark.parametrize(
    "replacement",
    [
        "  High-Value: jailbound_o_plus\n  Safety-Sensitivity: jailbound_o_minus",
        "  High-Value: arbitrary_minus\n  Safety-Sensitivity: arbitrary_plus",
    ],
)
def test_load_locked_inputs_rejects_noncanonical_branch_mapping(tmp_path: Path, replacement: str) -> None:
    module = _load_script()
    config = _study(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  High-Value: jailbound_o_minus\n  Safety-Sensitivity: jailbound_o_plus",
            replacement,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact branch mapping"):
        module.load_locked_inputs(config)


def test_load_locked_inputs_marks_missing_state_as_failed_unit(tmp_path: Path) -> None:
    module = _load_script()
    config = _study(tmp_path)
    (tmp_path / "matched/optimization/harmbench/jailbound_o_plus/states/100.pt").unlink()

    locked = module.load_locked_inputs(config)

    failed = next(unit for unit in locked.units if unit.branch is module.Branch.safety_sensitivity)
    assert failed.final_state_path is None
    assert failed.final_state_sha256 is None
    assert "missing" in failed.error


def test_load_locked_inputs_rejects_hidden_size_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    config = _study(tmp_path, hidden_size=3)
    config.write_text(config.read_text(encoding="utf-8").replace("hidden_size: 3", "hidden_size: 2"), encoding="utf-8")

    with pytest.raises(ValueError, match="hidden size"):
        module.load_locked_inputs(config)


def test_project_with_position_cosines_projects_both_editable_blocks() -> None:
    module = _load_script()
    vocabulary = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    audit = module.project_with_position_cosines(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
        vocabulary,
        chunk_size=2,
    )

    assert audit.prefix_token_ids == (0,)
    assert audit.seed_token_ids == (1,)
    assert audit.cosines == pytest.approx((1.0, 1.0))


class _CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = f"SYS\nUSER:{messages[-1]['content']}\nASSISTANT:"
        if not tokenize:
            return rendered
        return torch.tensor([[ord(char) for char in rendered]])

    def __call__(self, text, *, return_tensors=None, add_special_tokens=False, return_offsets_mapping=False):
        ids = torch.tensor([[ord(char) for char in text]])
        payload = {"input_ids": ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = torch.tensor([[[index, index + 1] for index in range(len(text))]])
        return payload

    def decode(self, token_ids, *, skip_special_tokens=True):
        return "".join(chr(token_id) for token_id in token_ids)


class _EmbeddingModel:
    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(256, 2)

    def get_input_embeddings(self):
        return self.embedding


def test_build_continuous_chat_input_inserts_state_inside_shared_chat_scaffold() -> None:
    module = _load_script()
    tokenizer = _CharacterTokenizer()
    model = _EmbeddingModel()
    z = torch.tensor([[[10.0, 11.0]]])
    u = torch.tensor([[[12.0, 13.0]]])

    result = module.build_continuous_chat_input(model, tokenizer, attack_text="xy", z=z, u=u)

    assert result.content_token_ids == (ord("x"), ord("y"))
    assert result.inputs_embeds.shape[1] == len("SYS\nUSER:xy\nASSISTANT:") + 2
    assert torch.equal(result.inputs_embeds[:, result.content_start, :], z[:, 0, :])
    assert torch.equal(result.inputs_embeds[:, result.content_start + 3, :], u[:, 0, :])
    assert result.attention_mask.shape == result.inputs_embeds.shape[:2]


def test_materialized_roundtrip_keeps_decode_retokenize_mismatch() -> None:
    module = _load_script()

    class NormalizingTokenizer(_CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[ord(char) for char in text.replace(" ", "")]])}

    audit = module.materialized_roundtrip(NormalizingTokenizer(), (ord("a"), ord(" "), ord("b")))

    assert audit.materialized_text == "a b"
    assert audit.retokenized_token_ids == (ord("a"), ord("b"))
    assert audit.roundtrip_exact_match is False


def test_materialized_roundtrip_decodes_full_vocabulary_without_dropping_special_tokens() -> None:
    module = _load_script()

    class SpecialTokenizer:
        def __init__(self) -> None:
            self.decode_calls: list[tuple[tuple[int, ...], bool]] = []

        def decode(self, token_ids, *, skip_special_tokens=True):
            self.decode_calls.append((tuple(token_ids), skip_special_tokens))
            return ("<SPECIAL>" if not skip_special_tokens else "") + "a"

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[0, ord("a")]])}

    tokenizer = SpecialTokenizer()
    audit = module.materialized_roundtrip(tokenizer, (0, ord("a")))

    assert tokenizer.decode_calls == [((0, ord("a")), False)]
    assert audit.materialized_text == "<SPECIAL>a"
    assert audit.retokenized_token_ids == (0, ord("a"))
    assert audit.roundtrip_exact_match is True


def _ready_run_unit(tmp_path: Path, module: ModuleType):
    config = _study(tmp_path)
    method_root = tmp_path / "matched/optimization/harmbench/jailbound_o_minus/states"
    torch.save({"z": torch.tensor([[[-1.0, 0.0]]]), "u": torch.tensor([[[0.0, -1.0]]])}, method_root / "0.pt")
    torch.save({"z": torch.tensor([[[1.0, 0.0]]]), "u": torch.tensor([[[0.0, 1.0]]])}, method_root / "100.pt")
    locked = module.load_locked_inputs(config)
    unit = next(candidate for candidate in locked.units if candidate.branch is module.Branch.high_value)
    model = _EmbeddingModel()
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[1] = torch.tensor([1.0, 0.0])
        model.embedding.weight[2] = torch.tensor([0.0, 1.0])
        model.embedding.weight[3] = torch.tensor([-1.0, 0.0])
        model.embedding.weight[4] = torch.tensor([0.0, -1.0])
    return locked, unit, model


def test_run_unit_uses_one_final_state_for_both_conditions_and_keeps_mismatch_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    locked, unit, model = _ready_run_unit(tmp_path, module)

    class MismatchTokenizer(_CharacterTokenizer):
        def __init__(self) -> None:
            self.decode_calls: list[tuple[tuple[int, ...], bool]] = []

        def decode(self, token_ids, *, skip_special_tokens=True):
            self.decode_calls.append((tuple(token_ids), skip_special_tokens))
            return super().decode(token_ids, skip_special_tokens=skip_special_tokens)

        def __call__(self, text, *, return_offsets_mapping=False, **kwargs):
            payload = super().__call__(text, return_offsets_mapping=return_offsets_mapping, **kwargs)
            if not return_offsets_mapping:
                payload["input_ids"] = payload["input_ids"][:, :-1]
            return payload

    tokenizer = MismatchTokenizer()
    calls: dict[str, object] = {}

    def fake_continuous(model_arg, tokenizer_arg, **kwargs):
        calls["continuous"] = (model_arg, tokenizer_arg, kwargs)
        return SimpleNamespace(response="continuous response")

    def fake_materialized(model_arg, tokenizer_arg, system_prompt, user_prompt, max_new_tokens):
        calls["materialized"] = (model_arg, tokenizer_arg, system_prompt, user_prompt, max_new_tokens)
        return SimpleNamespace(response="materialized response")

    monkeypatch.setattr(module, "generate_from_embeddings", fake_continuous)
    monkeypatch.setattr(module, "generate_one", fake_materialized)

    pair = module.run_unit(locked, unit, model=model, tokenizer=tokenizer, projection_chunk_size=32)

    content_ids = tuple(ord(char) for char in unit.attack_text)
    projected_ids = (1,) + content_ids + (2,)
    assert pair.status == "complete"
    assert pair.editable_projected_token_ids == (1, 2)
    assert pair.projected_token_ids == projected_ids
    assert pair.roundtrip_exact_match is False
    assert pair.retokenized_token_ids == projected_ids[:-1]
    assert tokenizer.decode_calls[-1] == (projected_ids, False)
    continuous_kwargs = calls["continuous"][2]
    assert continuous_kwargs["max_new_tokens"] == 512
    assert torch.equal(continuous_kwargs["inputs_embeds"][:, len("SYS\nUSER:"), :], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(
        continuous_kwargs["inputs_embeds"][:, len("SYS\nUSER:") + 1 : len("SYS\nUSER:") + 1 + len(content_ids), :],
        model.embedding(torch.tensor([content_ids])),
    )
    assert torch.equal(
        continuous_kwargs["inputs_embeds"][:, len("SYS\nUSER:") + 1 + len(content_ids), :],
        torch.tensor([[0.0, 1.0]]),
    )
    assert calls["materialized"][3] == pair.materialized_text
    assert calls["materialized"][4] == 512


def test_run_unit_converts_generation_exception_to_failed_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    locked, unit, model = _ready_run_unit(tmp_path, module)

    def fail_generation(*args, **kwargs):
        raise RuntimeError("fixture generation failure")

    monkeypatch.setattr(module, "generate_from_embeddings", fail_generation)

    pair = module.run_unit(locked, unit, model=model, tokenizer=_CharacterTokenizer(), projection_chunk_size=32)

    assert pair.status == "failed"
    assert pair.error == "pair generation error: RuntimeError: fixture generation failure"
    assert pair.judgments == {}


def test_dry_run_reports_locked_units_without_loading_model(tmp_path: Path, capsys) -> None:
    module = _load_script()
    config = _study(tmp_path)

    assert module.main(["--config", str(config), "--dry-run"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "checkpoint": 100,
        "failed_units": 0,
        "model_key": "qwen2_5_7b",
        "ready_units": 2,
        "sources": ["harmbench"],
        "total_units": 2,
    }
