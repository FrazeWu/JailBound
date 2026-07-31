from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, NoReturn, Self

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


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
        official_gbda_methods = [
            *expected_methods[:5],
            "gbda_official",
            *expected_methods[5:],
        ]
        expected_targets = ["qwen2_5_7b"]
        if self.data.sources != expected_sources or self.data.samples_per_source != 17:
            raise ValueError(
                "controlled data must use seven approved sources with 17 samples each"
            )
        if tuple(self.optimization.methods) not in {tuple(expected_methods), tuple(official_gbda_methods)}:
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


def load_config(path: str | Path) -> ExperimentConfig:
    payload: dict[str, Any] = OmegaConf.to_container(
        OmegaConf.load(path), resolve=True
    )
    return ExperimentConfig.model_validate(payload)


class H1V2StudyConfig(StrictModel):
    """Locked, independent confirmation settings for the FOL boundary study."""

    output_root: Path
    primary_judge_local_path: Path
    sources: list[str]
    candidate_count: int
    minimum_baseline_safe_count: int
    low: int
    middle: int
    high: int
    reserves: int
    accepted_directions: int
    max_direction_attempts: int
    radius_candidates: list[float]
    semantic_acceptance_floor: float
    relative_state_change_cap: float
    permutation_replicates: int
    bootstrap_replicates: int

    @field_validator("low", "middle", "high")
    @classmethod
    def _require_prime_display_band(cls, value: int) -> int:
        if value < 2 or any(value % divisor == 0 for divisor in range(2, int(value ** 0.5) + 1)):
            raise ValueError("H1-v2 endpoint-band counts must be prime")
        return value

    @model_validator(mode="after")
    def validate_confirmatory_protocol(self) -> "H1V2StudyConfig":
        if self.sources != ["jailbound", "s_eval"]:
            raise ValueError("H1-v2 is restricted to JailBound and S-Eval")
        if self.candidate_count != 81:
            raise ValueError("H1-v2 requires exactly 81 new candidates per source")
        if self.minimum_baseline_safe_count != 41:
            raise ValueError("H1-v2 requires 41 baseline-safe candidates before selection")
        if (self.low, self.middle, self.high) != (17, 3, 17):
            raise ValueError("H1-v2 requires the frozen 17/3/17 selection")
        if self.reserves != 4:
            raise ValueError("H1-v2 requires four ordered reserves")
        if self.accepted_directions != 32 or self.max_direction_attempts < self.accepted_directions:
            raise ValueError("H1-v2 requires 32 accepted directions and a sufficient attempt cap")
        radii = tuple(float(radius) for radius in self.radius_candidates)
        if len(radii) < 2 or any(radius <= 0.0 for radius in radii) or any(
            later <= earlier for earlier, later in zip(radii, radii[1:], strict=False)
        ):
            raise ValueError("H1-v2 radius candidates must be positive and strictly increasing")
        if not 0.0 < self.semantic_acceptance_floor <= 1.0:
            raise ValueError("H1-v2 semantic acceptance floor must be a probability")
        if self.relative_state_change_cap <= 0.0:
            raise ValueError("H1-v2 relative-state-change cap must be positive")
        if self.permutation_replicates < 999 or self.bootstrap_replicates < 999:
            raise ValueError("H1-v2 requires at least 999 permutation and bootstrap replicates")
        if self.output_root.name != "fol_h1_v2":
            raise ValueError("H1-v2 output root must end in fol_h1_v2")
        if self.primary_judge_local_path.name != "Octopus-SEval-14B":
            raise ValueError("H1-v2 primary judge path must identify Octopus-SEval-14B")
        return self


class H1V2Config(StrictModel):
    """Base controlled experiment plus isolated H1-v2 confirmation contract."""

    base: ExperimentConfig
    h1_v2: H1V2StudyConfig

    @model_validator(mode="after")
    def validate_isolation(self) -> "H1V2Config":
        if self.h1_v2.output_root == self.base.run.output_root:
            raise ValueError("H1-v2 output root must differ from the exploratory experiment root")
        return self


def load_h1_v2_config(path: str | Path) -> H1V2Config:
    """Load the separate H1-v2 protocol without loosening the base contract."""
    source = Path(path)
    payload = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("H1-v2 configuration must be a mapping")
    base_path = payload.pop("base_config", None)
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("H1-v2 configuration requires a base_config path")
    resolved_base = Path(base_path)
    if not resolved_base.is_absolute():
        resolved_base = source.parent / resolved_base
    payload["base"] = load_config(resolved_base).model_dump(mode="json")
    return H1V2Config.model_validate(payload)


class H1V3StudyConfig(StrictModel):
    """Strictly isolated settings for the H1-v3 local-radius extension."""

    source_root: Path
    output_root: Path
    primary_judge_local_path: Path
    sources: list[str]
    radius_candidates: list[float]
    accepted_directions: int = 32
    max_direction_attempts: int = 64
    semantic_acceptance_floor: float = 0.95
    relative_state_change_cap: float = 0.10
    permutation_replicates: int = 10_000
    bootstrap_replicates: int = 10_000

    @model_validator(mode="after")
    def validate_local_radius_extension(self) -> "H1V3StudyConfig":
        if self.sources != ["jailbound", "s_eval"]:
            raise ValueError("H1-v3 is restricted to JailBound and S-Eval")
        if tuple(float(value) for value in self.radius_candidates) != (0.4, 0.6):
            raise ValueError("H1-v3 requires exactly the new local radii [0.4, 0.6]")
        if self.accepted_directions != 32 or self.max_direction_attempts < 32:
            raise ValueError("H1-v3 requires 32 accepted directions and a sufficient attempt cap")
        if not 0.0 < self.semantic_acceptance_floor <= 1.0:
            raise ValueError("H1-v3 semantic acceptance floor must be a probability")
        if self.relative_state_change_cap != 0.10:
            raise ValueError("H1-v3 relative-state-change cap must remain 0.10")
        if self.permutation_replicates < 999 or self.bootstrap_replicates < 999:
            raise ValueError("H1-v3 requires at least 999 permutation and bootstrap replicates")
        if self.source_root.name != "fol_h1_v2":
            raise ValueError("H1-v3 source root must end in fol_h1_v2")
        if self.output_root.name != "fol_h1_v3":
            raise ValueError("H1-v3 output root must end in fol_h1_v3")
        if self.source_root == self.output_root:
            raise ValueError("H1-v3 source and output roots must differ")
        if self.primary_judge_local_path.name != "Octopus-SEval-14B":
            raise ValueError("H1-v3 primary judge path must identify Octopus-SEval-14B")
        return self


class H1V3Config(StrictModel):
    """Base controlled experiment plus the isolated H1-v3 extension contract."""

    base: ExperimentConfig
    h1_v3: H1V3StudyConfig

    @model_validator(mode="after")
    def validate_isolation(self) -> "H1V3Config":
        if self.h1_v3.output_root == self.base.run.output_root:
            raise ValueError("H1-v3 output root must differ from the exploratory experiment root")
        return self


def load_h1_v3_config(path: str | Path) -> H1V3Config:
    """Load the H1-v3 follow-up without permitting writes to H1-v2."""
    source = Path(path)
    payload = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("H1-v3 configuration must be a mapping")
    base_path = payload.pop("base_config", None)
    if not isinstance(base_path, str) or not base_path:
        raise ValueError("H1-v3 configuration requires a base_config path")
    resolved_base = Path(base_path)
    if not resolved_base.is_absolute():
        resolved_base = source.parent / resolved_base
    payload["base"] = load_config(resolved_base).model_dump(mode="json")
    return H1V3Config.model_validate(payload)
