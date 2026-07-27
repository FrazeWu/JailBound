"""Paired, prompt-level comparisons for newly executed result rows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
import math
import random
from typing import Iterable, Mapping

from .metrics import Rate


@dataclass(frozen=True)
class PairedRow:
    sample_id: str | None
    baseline_unsafe: bool
    method_unsafe: bool
    provenance: str = "new_run"


@dataclass(frozen=True)
class PairedAsrDifference:
    delta: float
    denominator: int
    discordant_method_only: int
    discordant_baseline_only: int


def paired_asr_difference(baseline: Iterable[PairedRow], method: Iterable[PairedRow]) -> PairedAsrDifference:
    baseline_rows = tuple(baseline)
    method_rows = tuple(method)
    rows = baseline_rows + method_rows
    if any(row.provenance == "frozen_pdf" or row.sample_id is None for row in rows):
        raise ValueError("frozen aggregates have no paired sample ids")
    left = {row.sample_id: row for row in baseline_rows}
    right = {row.sample_id: row for row in method_rows}
    if set(left) != set(right):
        raise ValueError("paired comparisons require identical sample ids")
    denominator = len(left)
    if not denominator:
        raise ValueError("paired comparisons require at least one sample")
    method_only = sum(not left[key].method_unsafe and right[key].method_unsafe for key in left)
    baseline_only = sum(left[key].method_unsafe and not right[key].method_unsafe for key in left)
    return PairedAsrDifference((sum(row.method_unsafe for row in right.values()) - sum(row.method_unsafe for row in left.values())) / denominator, denominator, method_only, baseline_only)


def exact_mcnemar_pvalue(*, method_only: int, baseline_only: int) -> float:
    """Return the two-sided exact McNemar p-value from discordant pair counts."""
    if method_only < 0 or baseline_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = method_only + baseline_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(method_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


@dataclass(frozen=True)
class JudgePair:
    """Two binary judge labels joined by response identity."""

    response_id: str
    primary_unsafe: bool
    secondary_unsafe: bool
    provenance: str = "new_run"


@dataclass(frozen=True)
class JudgeAgreement:
    agreement: float
    disagreement: float
    kappa: float | None
    denominator: int


def judge_agreement(pairs: Iterable[JudgePair]) -> JudgeAgreement:
    """Compute dependency-free raw agreement and Cohen's kappa for new rows."""
    values = tuple(pairs)
    if not values:
        raise ValueError("judge agreement requires at least one paired response")
    if any(pair.provenance == "frozen_pdf" for pair in values):
        raise ValueError("frozen_pdf judgments are context-only and cannot be compared")
    response_ids = [pair.response_id for pair in values]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("judge agreement requires unique response ids")
    denominator = len(values)
    agreement = sum(pair.primary_unsafe is pair.secondary_unsafe for pair in values) / denominator
    primary_positive = sum(pair.primary_unsafe for pair in values) / denominator
    secondary_positive = sum(pair.secondary_unsafe for pair in values) / denominator
    expected = primary_positive * secondary_positive + (1.0 - primary_positive) * (1.0 - secondary_positive)
    kappa = 1.0 if expected == 1.0 and agreement == 1.0 else (agreement - expected) / (1.0 - expected)
    return JudgeAgreement(agreement, 1.0 - agreement, kappa, denominator)


@dataclass(frozen=True)
class ThresholdRanking:
    """Exact-rate method ranks at one judge threshold."""

    threshold: float
    ranks: tuple[tuple[str, int], ...]


def _compare_rates(left: tuple[str, Rate], right: tuple[str, Rate]) -> int:
    left_name, left_rate = left
    right_name, right_rate = right
    if not left_rate.denominator or not right_rate.denominator:
        raise ValueError("threshold ranks require non-empty rate denominators")
    cross_product = left_rate.numerator * right_rate.denominator - right_rate.numerator * left_rate.denominator
    if cross_product:
        return -1 if cross_product > 0 else 1
    return (left_name > right_name) - (left_name < right_name)


def _rates_equal(left: Rate, right: Rate) -> bool:
    return left.numerator * right.denominator == right.numerator * left.denominator


def threshold_rank_sensitivity(rates_by_threshold: Mapping[float, Mapping[str, Rate]]) -> tuple[ThresholdRanking, ...]:
    """Rank methods at each threshold, preserving exact ties before display rounding."""
    if not rates_by_threshold:
        raise ValueError("threshold sensitivity requires at least one threshold")
    expected_methods: set[str] | None = None
    rankings: list[ThresholdRanking] = []
    for threshold in sorted(rates_by_threshold):
        rates = rates_by_threshold[threshold]
        if not rates:
            raise ValueError("each threshold requires at least one method rate")
        method_names = set(rates)
        if expected_methods is None:
            expected_methods = method_names
        elif method_names != expected_methods:
            raise ValueError("threshold sensitivity requires identical method sets")
        ordered = sorted(rates.items(), key=cmp_to_key(_compare_rates))
        ranks: list[tuple[str, int]] = []
        previous_rate: Rate | None = None
        rank = 0
        for position, (method, rate) in enumerate(ordered, start=1):
            if previous_rate is None or not _rates_equal(rate, previous_rate):
                rank = position
            ranks.append((method, rank))
            previous_rate = rate
        rankings.append(ThresholdRanking(float(threshold), tuple(ranks)))
    return tuple(rankings)


@dataclass(frozen=True)
class BootstrapPrompt:
    """One prompt-level bootstrap unit; all attached outcomes travel with its ID."""

    prompt_id: str
    stratum: str
    provenance: str = "new_run"


def bootstrap_prompt_ids(
    prompts: Iterable[BootstrapPrompt], *, replicates: int, seed: int
) -> tuple[tuple[str, ...], ...]:
    """Resample prompt IDs with replacement inside each fixed stratum."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    values = tuple(prompts)
    if any(prompt.provenance == "frozen_pdf" for prompt in values):
        raise ValueError("frozen_pdf prompts are context-only and cannot be bootstrapped")
    prompt_ids = [prompt.prompt_id for prompt in values]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("bootstrap requires one row per prompt id")
    groups: dict[str, tuple[str, ...]] = {}
    for stratum in sorted({prompt.stratum for prompt in values}):
        groups[stratum] = tuple(sorted(prompt.prompt_id for prompt in values if prompt.stratum == stratum))
    generator = random.Random(seed)
    draws: list[tuple[str, ...]] = []
    for _ in range(replicates):
        draw: list[str] = []
        for stratum in sorted(groups):
            members = groups[stratum]
            draw.extend(members[generator.randrange(len(members))] for _ in members)
        draws.append(tuple(draw))
    return tuple(draws)


@dataclass(frozen=True)
class FolClaimEvidence:
    """Pre-registered FOL hypothesis outcomes and quality-gate inputs."""

    h1: bool
    h2: bool
    h3: bool
    h4: bool
    secondary_same_direction: bool
    valid_paths: int
    usable_fraction: float
    band_acceptance_difference: float
    h1_interval_width: float
    h2_interval_width: float


@dataclass(frozen=True)
class FolFlipPrediction:
    """One direction-level flip outcome with prompt-grouped numeric features."""

    prompt_id: str
    flipped: bool
    fol: float
    controls: tuple[float, ...]


@dataclass(frozen=True)
class GroupedFlipComparison:
    """Held-out prompt-group comparison of controls-only and controls-plus-FOL models."""

    held_out_rows: int
    controls_auroc: float
    fol_auroc: float
    delta_auroc: float
    controls_auprc: float
    fol_auprc: float
    delta_auprc: float
    controls_brier: float
    fol_brier: float
    delta_brier: float
    controls_ece: float
    fol_ece: float
    delta_ece: float


def grouped_flip_prediction_comparison(
    rows: Iterable[FolFlipPrediction], *, folds: int, seed: int
) -> GroupedFlipComparison:
    """Evaluate FOL incremental value with group-disjoint prompt folds.

    Multiple directions from a prompt never cross a train/test boundary.  The
    function deliberately exposes only numeric outcomes and never prompt text.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    values = tuple(rows)
    if folds < 2:
        raise ValueError("folds must be at least two")
    if not values:
        raise ValueError("flip prediction requires rows")
    control_width = len(values[0].controls)
    if control_width < 1:
        raise ValueError("flip prediction requires at least one control")
    if any(not row.prompt_id for row in values):
        raise ValueError("flip prediction prompt ids must be non-empty")
    if any(type(row.flipped) is not bool for row in values):
        raise TypeError("flip prediction labels must be booleans")
    if any(
        len(row.controls) != control_width
        or not math.isfinite(row.fol)
        or any(not math.isfinite(value) for value in row.controls)
        for row in values
    ):
        raise ValueError("flip prediction features must be finite and rectangular")
    groups = [row.prompt_id for row in values]
    if len(set(groups)) < folds:
        raise ValueError("folds cannot exceed the number of prompt groups")
    labels = [int(row.flipped) for row in values]
    if len(set(labels)) != 2:
        raise ValueError("flip prediction requires both outcome classes")
    controls = [list(row.controls) for row in values]
    with_fol = [list(row.controls) + [row.fol] for row in values]
    controls_scores = [0.0] * len(values)
    fol_scores = [0.0] * len(values)
    splitter = GroupKFold(n_splits=folds)
    for train, test in splitter.split(controls, labels, groups):
        train_labels = [labels[index] for index in train]
        test_labels = [labels[index] for index in test]
        if len(set(train_labels)) != 2 or len(set(test_labels)) != 2:
            raise ValueError("each held-out fold must contain both outcome classes")
        controls_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=seed),
        )
        fol_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", random_state=seed),
        )
        controls_model.fit([controls[index] for index in train], train_labels)
        fol_model.fit([with_fol[index] for index in train], train_labels)
        for index, score in zip(
            test,
            controls_model.predict_proba([controls[item] for item in test])[:, 1],
            strict=True,
        ):
            controls_scores[index] = float(score)
        for index, score in zip(
            test,
            fol_model.predict_proba([with_fol[item] for item in test])[:, 1],
            strict=True,
        ):
            fol_scores[index] = float(score)

    def ece(scores: list[float], *, bins: int = 10) -> float:
        total = len(scores)
        return sum(
            (sum(1 for score in scores if lower <= score < upper) / total)
            * abs(
                sum(labels[index] for index, score in enumerate(scores) if lower <= score < upper)
                / sum(1 for score in scores if lower <= score < upper)
                - sum(score for score in scores if lower <= score < upper)
                / sum(1 for score in scores if lower <= score < upper)
            )
            for bin_index in range(bins)
            for lower, upper in [(bin_index / bins, (bin_index + 1) / bins if bin_index < bins - 1 else 1.0000001)]
            if any(lower <= score < upper for score in scores)
        )

    control_auroc = float(roc_auc_score(labels, controls_scores))
    fol_auroc = float(roc_auc_score(labels, fol_scores))
    control_auprc = float(average_precision_score(labels, controls_scores))
    fol_auprc = float(average_precision_score(labels, fol_scores))
    control_brier = float(brier_score_loss(labels, controls_scores))
    fol_brier = float(brier_score_loss(labels, fol_scores))
    control_ece = ece(controls_scores)
    fol_ece = ece(fol_scores)
    return GroupedFlipComparison(
        len(values),
        control_auroc,
        fol_auroc,
        fol_auroc - control_auroc,
        control_auprc,
        fol_auprc,
        fol_auprc - control_auprc,
        control_brier,
        fol_brier,
        fol_brier - control_brier,
        control_ece,
        fol_ece,
        fol_ece - control_ece,
    )


def decide_fol_claim(evidence: FolClaimEvidence) -> str:
    """Return the sole permitted FOL conclusion under the locked claim ladder."""
    if (
        evidence.valid_paths < 5
        or evidence.usable_fraction < 0.8
        or evidence.band_acceptance_difference > 0.15
        or evidence.h1_interval_width > 0.30
        or evidence.h2_interval_width > 0.50
    ):
        return "inconclusive"
    if evidence.h1 and evidence.h2 and evidence.h3 and evidence.secondary_same_direction:
        return "boundary_proxy_support"
    if evidence.h1 and evidence.secondary_same_direction:
        return "local_sensitivity_support_only"
    return "no_boundary_support"
