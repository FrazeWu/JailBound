from __future__ import annotations

import json
import operator
import pickle
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmark.reviewer_eval.config import ExperimentConfig, load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmark/reviewer_additions.yaml"
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


def test_default_pytest_collection_includes_reviewer_eval() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert project["tool"]["pytest"]["ini_options"]["testpaths"] == [
        "tests/corpora",
        "tests/test_python_sources_compile.py",
        "tests/reviewer_eval",
    ]
