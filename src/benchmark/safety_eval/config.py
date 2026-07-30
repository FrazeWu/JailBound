from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping, NoReturn, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_MODEL_COPY_UPDATE_UNSET = object()


class FrozenList(list[Any]):
    def _reject_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("frozen list cannot be modified")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[Any, tuple[list[Any]]]:
        return (_restore_frozen_list, (list(self),))


class FrozenDict(dict[Any, Any]):
    def _reject_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("frozen dict cannot be modified")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[Any, tuple[dict[Any, Any]]]:
        return (_restore_frozen_dict, (dict(self),))


def _restore_frozen_list(values: list[Any]) -> FrozenList:
    return FrozenList(values)


def _restore_frozen_dict(values: dict[Any, Any]) -> FrozenDict:
    return FrozenDict(values)


def _freeze_nested(value: Any) -> Any:
    if isinstance(value, (FrozenList, FrozenDict)):
        return value
    if isinstance(value, list):
        return FrozenList(_freeze_nested(item) for item in value)
    if isinstance(value, dict):
        return FrozenDict(
            (key, _freeze_nested(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_freeze_nested(item) for item in value)
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_post_init(self, __context: Any) -> None:
        for field_name in type(self).model_fields:
            object.__setattr__(
                self,
                field_name,
                _freeze_nested(getattr(self, field_name)),
            )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | object = _MODEL_COPY_UPDATE_UNSET,
        deep: bool = False,
    ) -> Self:
        if update is not _MODEL_COPY_UPDATE_UNSET:
            raise TypeError(
                "StrictModel.model_copy does not accept update; "
                "use model_dump() and model_validate()"
            )
        return super().model_copy(deep=deep)


class RunConfig(StrictModel):
    schema_version: Literal["reviewer_eval.v1"]
    seed: Literal[20260725]
    output_root: Path
    locked_config_name: str
    attention_implementation: Literal["eager"]


class V2RunConfig(StrictModel):
    schema_version: Literal["reviewer_eval.v2"]
    seed: Literal[20260725]
    output_root: Path
    locked_config_name: str
    attention_implementation: Literal["eager"]


class DataConfig(StrictModel):
    sources: list[str]
    samples_per_source: int
    paths: dict[str, Path]
    harmbench_targets_path: Path
    exclude_harmbench_functional_categories: list[str]


class ModelSpec(StrictModel):
    key: str
    repo_id: str
    local_path: Path | None = None
    revision: str | None = None
    dtype: Literal["bfloat16"]
    device_map: Literal["balanced", "single_gpu"]
    replicas: int


class ModelsConfig(StrictModel):
    surrogate: ModelSpec
    targets: list[ModelSpec]
    semantic_encoder: ModelSpec
    octopus: ModelSpec


class AnnotationConfig(StrictModel):
    model: str
    revision: str
    endpoint: str
    template_path: Path
    confidence_artifact: Path
    temperature: float = Field(strict=True)
    repair_attempts: int = Field(strict=True)

    @field_validator("model", "revision")
    @classmethod
    def validate_non_empty_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("annotation model and revision must be non-empty")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("annotation temperature must be exactly 0.0")
        return value

    @field_validator("repair_attempts")
    @classmethod
    def validate_repair_attempts(cls, value: int) -> int:
        if value != 1:
            raise ValueError("annotation repair_attempts must be exactly 1")
        return value


class OptimizationConfig(StrictModel):
    methods: list[str]
    update_budget: int
    dual_branch_updates: dict[str, int]
    checkpoints: list[int]
    prefix_tokens: int
    editable_seed_tokens: int
    candidate_cap: int
    learning_rate: float
    gbda_learning_rate: float
    gcg_search_width: int
    lambda_fol: float
    epsilon: float
    gamma_z: float
    gamma_u: float
    grad_clip: float
    answer_anchors: list[str]
    refusal_anchors: list[str]


class PrefixInitializationConfig(StrictModel):
    strategy: Literal["repeat_token"]
    token_text: str = Field(min_length=1)


class V2OptimizationConfig(StrictModel):
    methods: list[str]
    update_budget: int
    dual_branch_updates: dict[str, int]
    checkpoints: list[int]
    prefix_tokens: int
    prefix_initialization: PrefixInitializationConfig
    final_states_per_branch: int = Field(ge=1)
    anchor_set_version: str = Field(min_length=1)
    candidate_cap: int
    learning_rate: float
    gbda_learning_rate: float
    gcg_search_width: int
    lambda_fol: float
    epsilon: float
    gamma_z: float
    gamma_u: float
    grad_clip: float
    answer_anchors: list[str]
    refusal_anchors: list[str]

    @field_validator("anchor_set_version")
    @classmethod
    def validate_anchor_set_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("anchor_set_version must be non-empty")
        return value


class SemanticConfig(StrictModel):
    calibration_examples_per_source: int
    target_positive_recall: float
    threshold_artifact: Path
    category_consistency: bool
    max_mutation_attempts: int


class JudgeSpec(StrictModel):
    key: str
    threshold: float
    threshold_offsets: list[float]
    model: str | None = None
    endpoint: str | None = None
    temperature: float = 0.0


class JudgingConfig(StrictModel):
    primary: JudgeSpec
    secondary: JudgeSpec
    max_new_tokens: int
    author_audit_size: int


class FolConfig(StrictModel):
    sources: list[str]
    validation_per_source: int
    radius_calibration_per_source: int
    low: int
    middle: int
    high: int
    base_radius_candidates: list[float]
    radius_multipliers: list[float]
    directions_per_radius: int
    max_direction_attempts: int
    minimum_accepted_directions: int
    micro_noise_multiplier: float
    hvp_directions: int
    interpolation_points: int
    minimum_valid_interpolation_points: int
    minimum_valid_paths: int
    bootstrap_replicates: int
    permutation_replicates: int


class ExperimentConfig(StrictModel):
    run: RunConfig
    data: DataConfig
    models: ModelsConfig
    optimization: OptimizationConfig
    semantic: SemanticConfig
    judging: JudgingConfig
    fol: FolConfig

    @model_validator(mode="after")
    def validate_approved_scope(self) -> "ExperimentConfig":
        expected_sources = [
            "advbench",
            "harmbench",
            "safetybench",
            "sg_bench",
            "jailbreakbench",
            "jailbound",
            "s_eval",
        ]
        expected_methods = [
            "init",
            "random_mutation",
            "zol",
            "pez",
            "gbda",
            "gcg",
            "jailbound_o_minus",
            "jailbound_o_plus",
            "dual_branch",
        ]
        expected_targets = ["qwen2_5_7b"]
        if self.data.sources != expected_sources or self.data.samples_per_source != 17:
            raise ValueError(
                "controlled data must use seven approved sources with 17 samples each"
            )
        if self.optimization.methods != expected_methods:
            raise ValueError("optimization method identities or order changed")
        if self.optimization.update_budget != 100:
            raise ValueError("single-branch update budget must be 100")
        if self.optimization.dual_branch_updates != {"o_minus": 50, "o_plus": 50}:
            raise ValueError("dual branch must consume exactly 100 updates as 50/50")
        if self.optimization.checkpoints != [0, 25, 50, 100]:
            raise ValueError("checkpoint policy must be 0/25/50/100")
        if [model.key for model in self.models.targets] != expected_targets:
            raise ValueError("target models must remain in the approved serial order")
        if self.models.surrogate.key != "qwen2_5_7b":
            raise ValueError("white-box surrogate must be the local Qwen2.5-7B-Instruct")
        primary = self.judging.primary
        if (
            primary.key,
            primary.model,
            primary.endpoint,
            primary.temperature,
        ) != ("octopus_seval_14b", None, None, 0.0):
            raise ValueError("primary judge must remain the approved local Octopus definition")
        secondary = self.judging.secondary
        if (
            secondary.key,
            secondary.model,
            secondary.endpoint,
            secondary.temperature,
        ) != (
            "qwen32_compat",
            "qwen3-32b-awq",
            "http://localhost:8001/v1",
            0.0,
        ):
            raise ValueError("secondary judge must remain the approved local Qwen compatibility endpoint")
        if (
            self.fol.validation_per_source,
            self.fol.low,
            self.fol.middle,
            self.fol.high,
        ) != (17, 7, 3, 7):
            raise ValueError(
                "FOL validation must retain the approved 7/3/7 split within 17"
            )
        if self.fol.sources != ["jailbound", "s_eval"]:
            raise ValueError(
                "focused FOL validation is restricted to JailBound and S-Eval"
            )
        return self


class V2ExperimentConfig(ExperimentConfig):
    run: V2RunConfig
    optimization: V2OptimizationConfig
    annotation: AnnotationConfig


def load_config(path: str | Path) -> ExperimentConfig:
    payload: dict[str, Any] = OmegaConf.to_container(
        OmegaConf.load(path), resolve=True
    )
    return ExperimentConfig.model_validate(payload)


def _contains_v1_schema_version(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("schema_version") == "reviewer_eval.v1":
            return True
        return any(_contains_v1_schema_version(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_v1_schema_version(item) for item in value)
    return False


def _reject_structured_v1_artifacts(config: V2ExperimentConfig) -> None:
    artifact_names = {
        "locked_config.json",
        config.run.locked_config_name,
        "run_manifest.json",
        "optimization_config.json",
    }
    for artifact_name in sorted(artifact_names):
        artifact = config.run.output_root / artifact_name
        if not artifact.is_file():
            continue
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if _contains_v1_schema_version(payload):
            raise ValueError(
                f"v2 output root contains a structured reviewer_eval.v1 artifact: {artifact}"
            )


def load_v2_config(path: str | Path) -> V2ExperimentConfig:
    payload: dict[str, Any] = OmegaConf.to_container(
        OmegaConf.load(path), resolve=True
    )
    config = V2ExperimentConfig.model_validate(payload)
    _reject_structured_v1_artifacts(config)
    return config
