from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    schema_version: Literal["reviewer_eval.v1"]
    seed: Literal[20260725]
    output_root: Path
    locked_config_name: str


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
    wildguard: ModelSpec


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
        expected_targets = [
            "qwen2_5_14b",
            "qwen2_5_7b",
            "llama3_1_8b",
            "gemma2_9b",
        ]
        if self.data.sources != expected_sources or self.data.samples_per_source != 50:
            raise ValueError(
                "controlled data must use seven approved sources with 50 samples each"
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
        if self.models.surrogate.key != "qwen2_5_14b":
            raise ValueError("white-box surrogate must be Qwen2.5-14B-Instruct")
        if (
            self.fol.validation_per_source,
            self.fol.low,
            self.fol.middle,
            self.fol.high,
        ) != (31, 11, 9, 11):
            raise ValueError(
                "FOL validation must retain the approved 11/9/11 split within 31"
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
