from __future__ import annotations

import json
import operator
import pickle
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest
from omegaconf import OmegaConf

from benchmark.safety_eval.config import (
    ExperimentConfig,
    V2ExperimentConfig,
    load_config,
    load_v2_config,
    structured_artifact_contains_v1,
)
from benchmark.safety_eval.schema import ComputeCounters, OptimizationRecord, RecordStatus


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/safety_eval_additions.yaml"
V2_CONFIG = ROOT / "configs/benchmark/safety_eval_paper_v2.yaml"
PYPROJECT = ROOT / "pyproject.toml"

Mutation = Callable[[Any], Any]

LIST_MUTATIONS: list[tuple[str, Mutation]] = [
    ("setitem", lambda value: operator.setitem(value, 0, 100)),
    ("delitem", lambda value: operator.delitem(value, 0)),
    ("iadd", lambda value: operator.iadd(value, [100])),
    ("imul", lambda value: operator.imul(value, 2)),
    ("append", lambda value: value.append(100)),
    ("clear", lambda value: value.clear()),
    ("extend", lambda value: value.extend([100])),
    ("insert", lambda value: value.insert(0, 100)),
    ("pop", lambda value: value.pop()),
    ("remove", lambda value: value.remove(0)),
    ("reverse", lambda value: value.reverse()),
    ("sort", lambda value: value.sort()),
]

DICT_MUTATIONS: list[tuple[str, Mutation]] = [
    ("setitem", lambda value: operator.setitem(value, "o_plus", 51)),
    ("delitem", lambda value: operator.delitem(value, "o_plus")),
    ("ior", lambda value: operator.ior(value, {"o_plus": 51})),
    ("clear", lambda value: value.clear()),
    ("pop", lambda value: value.pop("o_plus")),
    ("popitem", lambda value: value.popitem()),
    ("setdefault", lambda value: value.setdefault("other", 1)),
    ("update", lambda value: value.update({"o_plus": 51})),
]


def _v2_payload() -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(V2_CONFIG), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _write_v2_config(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "v2.yaml"
    OmegaConf.save(config=OmegaConf.create(payload), f=path)
    return path


def test_checked_in_v2_config_loads_strict_paper_contract() -> None:
    config = load_v2_config(V2_CONFIG)

    assert isinstance(config, V2ExperimentConfig)
    assert not isinstance(config, ExperimentConfig)
    assert config.run.schema_version == "reviewer_eval.v2"
    assert config.run.output_root == Path("outputs/results/reviewer_eval_v2")
    assert config.annotation.model == "de-aligned-annotator"
    assert config.annotation.revision == "immutable-revision"
    assert config.annotation.temperature == 0.0
    assert config.annotation.repair_attempts == 1
    assert config.optimization.prefix_tokens == 20
    assert config.optimization.prefix_initialization.strategy == "repeat_token"
    assert config.optimization.prefix_initialization.token_text == "!"


def test_v2_smoke_mode_requires_an_isolated_one_sample_output(tmp_path: Path) -> None:
    payload = _v2_payload()
    payload["run"]["smoke_mode"] = True
    payload["run"]["output_root"] = "outputs/results/smoke/one-sample"
    payload["data"]["sources"] = ["advbench"]
    payload["data"]["samples_per_source"] = 1
    payload["optimization"]["methods"] = ["random_mutation"]

    config = load_v2_config(_write_v2_config(tmp_path, payload))
    assert config.run.smoke_mode is True

    payload["run"]["output_root"] = "outputs/results/reviewer_eval_v2"
    with pytest.raises(ValueError, match="smoke output root"):
        load_v2_config(_write_v2_config(tmp_path, payload))

    payload["run"]["output_root"] = "outputs/results/smoke/one-sample"
    payload["optimization"]["methods"] = ["dual_branch"]
    with pytest.raises(ValueError, match="does not support dual_branch"):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_v2_loader_requires_a_local_secondary_judge_snapshot(tmp_path: Path) -> None:
    payload = _v2_payload()
    payload["judging"]["secondary"]["local_path"] = "/fixture/qwen32"
    del payload["judging"]["secondary"]["local_path"]

    with pytest.raises(ValueError, match="secondary judge requires a local snapshot"):
        load_v2_config(_write_v2_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("optimization", "prefix_initialization", "strategy"), None, "strategy"),
        (("annotation", "temperature"), 0.1, "temperature"),
        (("annotation", "temperature"), 0, "temperature"),
        (("annotation", "temperature"), False, "temperature"),
        (("annotation", "repair_attempts"), 2, "repair_attempts"),
        (("annotation", "repair_attempts"), True, "repair_attempts"),
        (("optimization", "final_states_per_branch"), 0, "final_states_per_branch"),
        (("optimization", "anchor_set_version"), "", "anchor_set_version"),
        (("run", "schema_version"), "reviewer_eval.v1", "reviewer_eval.v2"),
    ],
    ids=[
        "missing-prefix-strategy",
        "nonzero-annotation-temperature",
        "integer-annotation-temperature",
        "boolean-annotation-temperature",
        "wrong-repair-attempts",
        "boolean-repair-attempts",
        "no-final-states",
        "empty-anchor-set-version",
        "v1-schema-version",
    ],
)
def test_v2_loader_rejects_invalid_contract_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _v2_payload()
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    if value is None:
        del target[path[-1]]
    else:
        target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_v2_config(_write_v2_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [("model", ""), ("revision", "")],
    ids=["empty-model", "empty-revision"],
)
def test_v2_loader_rejects_empty_annotation_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    payload = _v2_payload()
    payload["annotation"][field] = value

    with pytest.raises(ValueError, match=field):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_v2_loader_rejects_invalid_repeat_token_initialization(tmp_path: Path) -> None:
    payload = _v2_payload()
    payload["optimization"]["prefix_initialization"]["token_text"] = ""

    with pytest.raises(ValueError, match="token_text"):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_v2_loader_rejects_output_root_with_structured_v1_artifact(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    output_root.mkdir()
    (output_root / "locked_config.json").write_text(
        json.dumps({"run": {"schema_version": "reviewer_eval.v1"}}),
        encoding="utf-8",
    )
    payload = _v2_payload()
    payload["run"]["output_root"] = str(output_root)

    with pytest.raises(ValueError, match="structured reviewer_eval.v1 artifact"):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_structured_artifact_contains_v1_is_public(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"record": {"schema_version": "reviewer_eval.v1"}}),
        encoding="utf-8",
    )

    assert structured_artifact_contains_v1(artifact)


def test_v2_loader_cannot_hide_standard_v1_lock_with_renamed_config_lock(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    output_root.mkdir()
    (output_root / "locked_config.json").write_text(
        json.dumps({"run": {"schema_version": "reviewer_eval.v1"}}),
        encoding="utf-8",
    )
    payload = _v2_payload()
    payload["run"]["output_root"] = str(output_root)
    payload["run"]["locked_config_name"] = "v2_locked_config.json"

    with pytest.raises(ValueError, match="structured reviewer_eval.v1 artifact"):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_v2_loader_rejects_nested_structured_v1_jsonl_ledger(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    ledger = output_root / "optimization" / "advbench" / "pez" / "records.jsonl"
    ledger.parent.mkdir(parents=True)
    record = OptimizationRecord(
        schema_version="reviewer_eval.v1",
        run_id="run:legacy",
        config_hash="a" * 64,
        git_revision="1234567890abcdef",
        cell_id="cell:legacy",
        sample_id="advbench:000000:legacy",
        source="advbench",
        method="pez",
        checkpoint=25,
        random_seed=20260725,
        status=RecordStatus.complete,
        failure_kind=None,
        failure_reason=None,
        state_path=None,
        representation="token_ids",
        attack_loss=0.25,
        fol=0.5,
        internal_margin=0.1,
        materialized_prompt=None,
        counters=ComputeCounters(),
    )
    ledger.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    payload = _v2_payload()
    payload["run"]["output_root"] = str(output_root)

    with pytest.raises(ValueError, match="structured reviewer_eval.v1 artifact"):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_v2_loader_rejects_nested_uppercase_v1_jsonl_ledger(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "candidate"
    ledger = output_root / "optimization" / "records.JSONL"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "{}\n"
        + json.dumps({"record": {"schema_version": "reviewer_eval.v1"}})
        + "\n",
        encoding="utf-8",
    )
    payload = _v2_payload()
    payload["run"]["output_root"] = str(output_root)

    with pytest.raises(ValueError, match="structured reviewer_eval.v1 artifact"):
        load_v2_config(_write_v2_config(tmp_path, payload))


def test_v2_loader_does_not_guess_artifact_version_from_directory_name(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "reviewer_eval_v1_but_empty"
    output_root.mkdir()
    payload = _v2_payload()
    payload["run"]["output_root"] = str(output_root)

    config = load_v2_config(_write_v2_config(tmp_path, payload))

    assert config.run.output_root == output_root


def test_checked_in_config_has_approved_scope() -> None:
    config = load_config(CONFIG)
    assert config.run.output_root == Path(
        "outputs/results/reviewer_additions_n17_qwen7b_local_qwen32_compat_eager_randomfix"
    )
    assert config.run.attention_implementation == "eager"
    assert config.data.samples_per_source == 17
    assert tuple(config.data.sources) == (
        "advbench",
        "harmbench",
        "safetybench",
        "sg_bench",
        "jailbreakbench",
        "jailbound",
        "s_eval",
    )
    assert config.optimization.update_budget == 100
    assert config.optimization.dual_branch_updates == {"o_minus": 50, "o_plus": 50}
    assert config.optimization.checkpoints == [0, 25, 50, 100]
    assert [model.key for model in config.models.targets] == ["qwen2_5_7b"]
    assert config.models.surrogate.local_path == Path("/home/wh/models/qwen/Qwen2___5-7B-Instruct")
    assert config.fol.validation_per_source == 17
    assert (config.fol.low, config.fol.middle, config.fol.high) == (7, 3, 7)


def test_config_rejects_non_matched_dual_budget() -> None:
    payload = load_config(CONFIG).model_dump(mode="json")
    payload = deepcopy(payload)
    payload["optimization"]["dual_branch_updates"]["o_plus"] = 51
    with pytest.raises(ValueError, match="dual branch must consume exactly 100 updates"):
        ExperimentConfig.model_validate(payload)


def test_config_rejects_unapproved_fol_split() -> None:
    payload = load_config(CONFIG).model_dump(mode="json")
    payload["fol"]["low"] = 5
    payload["fol"]["middle"] = 5
    with pytest.raises(ValueError, match="7/3/7"):
        ExperimentConfig.model_validate(payload)


def test_config_rejects_non_octopus_primary_judge() -> None:
    payload = deepcopy(load_config(CONFIG).model_dump(mode="json"))
    payload["judging"]["primary"]["key"] = "other_judge"

    with pytest.raises(ValueError, match="primary judge must remain"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("judge_name", "field_name", "changed_value", "validation_message"),
    [
        ("primary", "model", "unexpected-model", "primary judge must remain"),
        ("primary", "endpoint", "http://example.invalid/v1", "primary judge must remain"),
        ("primary", "temperature", 0.1, "primary judge must remain"),
        ("secondary", "key", "other_judge", "secondary judge must remain"),
        ("secondary", "model", "other-model", "secondary judge must remain"),
        ("secondary", "endpoint", "http://example.invalid/v1", "secondary judge must remain"),
        ("secondary", "temperature", 0.1, "secondary judge must remain"),
    ],
    ids=[
        "primary-model",
        "primary-endpoint",
        "primary-temperature",
        "secondary-key",
        "secondary-model",
        "secondary-endpoint",
        "secondary-temperature",
    ],
)
def test_config_rejects_mutated_locked_judge_identity(
    judge_name: str,
    field_name: str,
    changed_value: object,
    validation_message: str,
) -> None:
    payload = deepcopy(load_config(CONFIG).model_dump(mode="json"))
    payload["judging"][judge_name][field_name] = changed_value

    with pytest.raises(ValueError, match=validation_message):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [mutation for _, mutation in LIST_MUTATIONS],
    ids=[name for name, _ in LIST_MUTATIONS],
)
def test_loaded_nested_lists_reject_common_mutations(mutation: Mutation) -> None:
    config = load_config(CONFIG)

    with pytest.raises(TypeError, match="frozen"):
        mutation(config.optimization.checkpoints)


@pytest.mark.parametrize(
    "mutation",
    [mutation for _, mutation in DICT_MUTATIONS],
    ids=[name for name, _ in DICT_MUTATIONS],
)
def test_loaded_nested_dicts_reject_common_mutations(mutation: Mutation) -> None:
    config = load_config(CONFIG)

    with pytest.raises(TypeError, match="frozen"):
        mutation(config.optimization.dual_branch_updates)


def test_json_dump_uses_ordinary_json_collections() -> None:
    config = load_config(CONFIG)
    payload = config.model_dump(mode="json")

    assert type(payload) is dict
    assert type(payload["optimization"]["checkpoints"]) is list
    assert type(payload["optimization"]["dual_branch_updates"]) is dict
    assert json.loads(config.model_dump_json()) == payload


def test_strict_model_rejects_model_copy_updates() -> None:
    config = load_config(CONFIG)

    with pytest.raises(TypeError, match="model_copy.*update"):
        config.model_copy(update={"optimization": {"update_budget": 101}})


def _assert_nested_containers_are_immutable(config: ExperimentConfig) -> None:
    with pytest.raises(TypeError, match="frozen"):
        config.optimization.checkpoints.append(101)
    with pytest.raises(TypeError, match="frozen"):
        config.optimization.dual_branch_updates["o_plus"] = 51


@pytest.mark.parametrize(
    "copy_config",
    [
        deepcopy,
        lambda config: config.model_copy(deep=True),
        lambda config: pickle.loads(pickle.dumps(config)),
    ],
    ids=["deepcopy", "model_copy_deep", "pickle_round_trip"],
)
def test_config_copy_paths_preserve_immutable_nested_containers(
    copy_config: Callable[[ExperimentConfig], ExperimentConfig],
) -> None:
    config = load_config(CONFIG)
    copied = copy_config(config)

    assert copied is not config
    assert copied == config
    assert copied.model_dump(mode="json") == config.model_dump(mode="json")
    _assert_nested_containers_are_immutable(copied)


def test_project_declares_uv_indexes_used_by_lockfile() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert project["tool"]["uv"]["index"] == [
        {
            "url": "https://pypi.tuna.tsinghua.edu.cn/simple",
            "default": True,
        },
        {"url": "https://mirrors.aliyun.com/pypi/simple/"},
        {"url": "https://repo.huaweicloud.com/repository/pypi/simple/"},
    ]


def test_default_pytest_collection_includes_safety_eval() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert project["tool"]["pytest"]["ini_options"]["testpaths"] == [
        "tests/corpora",
        "tests/test_python_sources_compile.py",
        "tests/safety_eval",
    ]
