"""Pure CPU tensor and deterministic selection utilities for FOL boundary studies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import math
import random
from statistics import median

import torch


LOW_BAND_SIZE = 7
MIDDLE_BAND_SIZE = 3
HIGH_BAND_SIZE = 7
INTERPOLATION_POINT_COUNT = 21
MINIMUM_VALID_INTERPOLATION_POINTS = 17
MINIMUM_VALID_INTERPOLATION_PATHS = 5


@dataclass(frozen=True)
class FolValue:
    """A neutral sample identifier paired with its scalar FOL value."""

    sample_id: str
    fol: float


@dataclass(frozen=True)
class FolBandSplit:
    """Disjoint sample IDs selected for the fixed low/middle/high FOL bands."""

    low: tuple[str, ...]
    middle: tuple[str, ...]
    high: tuple[str, ...]


@dataclass(frozen=True)
class BehaviorFlipRate:
    """A behavior-flip rate or an explicit acceptance-denominator failure."""

    rate: float | None
    accepted_count: int
    acceptance_failure: bool


@dataclass(frozen=True)
class FolPerturbationOutcome:
    """Content-free outcome for one scheduled local perturbation direction."""

    source: str
    sample_id: str
    band: str
    radius: float
    accepted: bool
    primary_label: bool
    primary_baseline_label: bool
    secondary_label: bool | None
    secondary_baseline_label: bool | None


@dataclass(frozen=True)
class FolBfrSummary:
    """Prompt-weighted behavior-flip summary for one band, radius, and judge."""

    source: str
    band: str
    radius: float
    judge_key: str
    prompt_count: int
    eligible_prompt_count: int
    sparse_prompt_count: int
    accepted_direction_count: int
    mean_bfr: float | None


@dataclass(frozen=True)
class InterpolationPoint:
    """Content-free measurements at one fixed position on an interpolation path."""

    path_id: str
    point_index: int
    semantic_accepted: bool
    unsafe_label: bool
    fol: float
    curvature: float


@dataclass(frozen=True)
class InterpolationPeakSummary:
    """H3 peak-distance evidence, retaining paths without an observed crossing."""

    valid_path_count: int
    crossing_path_count: int
    fol_mean_distance: float | None
    curvature_mean_distance: float | None
    random_mean_distance: float | None
    fol_vs_random_pvalue: float | None
    fol_vs_curvature_pvalue: float | None


@dataclass(frozen=True)
class LabeledEditableState:
    """One content-free editable-state candidate paired with an external label."""

    sample_id: str
    candidate_id: str
    unsafe_label: bool
    vector: torch.Tensor


@dataclass(frozen=True)
class OppositeLabelPair:
    """Nearest state pair whose labels differ, ordered safe then unsafe."""

    sample_id: str
    safe_candidate_id: str
    unsafe_candidate_id: str
    distance: float


@dataclass(frozen=True)
class MarginCalibrationRow:
    """One non-validation margin/behavior pair for boundary calibration."""

    sample_id: str
    margin: float
    unsafe_label: bool


@dataclass(frozen=True)
class MarginCalibration:
    """A reproducible monotone calibration with its eligible training IDs."""

    training_ids: tuple[str, ...]
    threshold: float | None
    brier_score: float
    curve: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MarginCrossing:
    """The bracket enclosing the first calibrated-margin crossing."""

    lower: float
    upper: float
    estimate: float


@dataclass(frozen=True)
class BehaviorDistance:
    """The behavior-flip distance or a right-censored lower bound."""

    estimate: float | None
    lower: float
    upper: float
    right_censored: bool


@dataclass(frozen=True)
class RightCensoredCoxEstimate:
    """One-covariate Cox estimate retaining right-censored distances."""

    sample_count: int
    event_count: int
    log_hazard_ratio: float
    hazard_ratio: float
    standard_error: float
    pvalue: float


def fit_right_censored_cox(
    *, covariates: Sequence[float], times: Sequence[float], events: Sequence[bool]
) -> RightCensoredCoxEstimate:
    """Fit a one-covariate Cox model using observed flips as events.

    Larger covariates with positive coefficients imply a higher flip hazard and
    therefore a smaller behavior-boundary distance.  Censored rows remain in
    every compatible risk set.
    """
    if len(covariates) != len(times) or len(times) != len(events) or len(times) < 3:
        raise ValueError("Cox fitting requires equally sized inputs with at least three rows")
    if any(not math.isfinite(float(value)) for value in (*covariates, *times)):
        raise ValueError("Cox covariates and times must be finite")
    if any(float(value) <= 0.0 for value in times):
        raise ValueError("Cox times must be positive")
    if any(type(event) is not bool for event in events):
        raise TypeError("Cox events must be booleans")
    if sum(events) < 2 or len(set(float(value) for value in covariates)) < 2:
        raise ValueError("Cox fitting requires at least two events and variable covariates")

    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import norm

    x = np.asarray(covariates, dtype=float)
    t = np.asarray(times, dtype=float)
    observed = np.asarray(events, dtype=bool)

    def negative_log_partial_likelihood(beta: np.ndarray) -> float:
        value = 0.0
        for index in np.flatnonzero(observed):
            risk = beta[0] * x[t >= t[index]]
            value -= beta[0] * x[index] - float(np.logaddexp.reduce(risk))
        return value

    fitted = minimize(negative_log_partial_likelihood, np.zeros(1), method="BFGS")
    if not fitted.success or fitted.hess_inv.shape != (1, 1):
        raise ValueError("Cox optimizer did not converge")
    variance = float(fitted.hess_inv[0, 0])
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("Cox optimizer returned an invalid variance")
    coefficient = float(fitted.x[0])
    standard_error = math.sqrt(variance)
    return RightCensoredCoxEstimate(
        sample_count=len(covariates),
        event_count=int(observed.sum()),
        log_hazard_ratio=coefficient,
        hazard_ratio=math.exp(coefficient),
        standard_error=standard_error,
        pvalue=float(2.0 * norm.sf(abs(coefficient / standard_error))),
    )


def estimate_behavior_distance(rates_by_radius: Mapping[float, float]) -> BehaviorDistance:
    """Fit an increasing BFR curve and retain unreached distances as censored."""
    from sklearn.isotonic import IsotonicRegression

    if len(rates_by_radius) < 2:
        raise ValueError("behavior distance requires at least two radii")
    radii = sorted(float(radius) for radius in rates_by_radius)
    if any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError("behavior-distance radii must be finite positive values")
    rates = [float(rates_by_radius[radius]) for radius in radii]
    if any(not math.isfinite(rate) or not 0.0 <= rate <= 1.0 for rate in rates):
        raise ValueError("behavior-flip rates must be finite probabilities")
    fitted = [float(value) for value in IsotonicRegression(increasing=True).fit_transform(radii, rates)]
    previous_radius: float | None = None
    previous_rate: float | None = None
    for radius, rate in zip(radii, fitted, strict=True):
        if rate >= 0.5:
            if previous_radius is None or previous_rate is None or rate == previous_rate:
                estimate = radius
            else:
                estimate = previous_radius + (0.5 - previous_rate) * (radius - previous_radius) / (rate - previous_rate)
            return BehaviorDistance(estimate, estimate, estimate, False)
        previous_radius, previous_rate = radius, rate
    return BehaviorDistance(None, radii[-1], math.inf, True)


def fit_margin_calibration(
    rows: Sequence[MarginCalibrationRow], *, excluded_ids: set[str] | frozenset[str]
) -> MarginCalibration:
    """Fit increasing isotonic behavior calibration outside validation IDs."""
    from sklearn.isotonic import IsotonicRegression

    training = tuple(
        sorted((row for row in rows if row.sample_id not in excluded_ids), key=lambda row: row.sample_id)
    )
    if len(training) < 2:
        raise ValueError("margin calibration requires at least two non-validation rows")
    if len({row.sample_id for row in training}) != len(training):
        raise ValueError("margin calibration sample IDs must be unique")
    if any(not math.isfinite(row.margin) for row in training):
        raise ValueError("margin calibration values must be finite")
    labels = [float(row.unsafe_label) for row in training]
    if len(set(labels)) != 2:
        raise ValueError("margin calibration requires both behavior labels")

    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    margins = [row.margin for row in training]
    predicted = [float(value) for value in model.fit_transform(margins, labels)]
    curve = tuple(sorted(zip(margins, predicted, strict=True)))
    threshold: float | None = None
    previous_margin: float | None = None
    previous_probability: float | None = None
    for margin, probability in curve:
        if probability >= 0.5:
            if previous_margin is None or previous_probability is None or probability == previous_probability:
                threshold = margin
            else:
                fraction = (0.5 - previous_probability) / (probability - previous_probability)
                threshold = previous_margin + fraction * (margin - previous_margin)
            break
        previous_margin, previous_probability = margin, probability
    brier = sum((probability - label) ** 2 for probability, label in zip(predicted, labels, strict=True)) / len(training)
    return MarginCalibration(
        training_ids=tuple(row.sample_id for row in training),
        threshold=threshold,
        brier_score=brier,
        curve=curve,
    )


def locate_margin_crossing(
    evaluate_margin: Callable[[float], float],
    *,
    threshold: float,
    lower: float,
    upper: float,
    iterations: int = 12,
) -> MarginCrossing:
    """Binarily isolate a crossing bracket in an already-valid radius interval."""
    if not all(math.isfinite(value) for value in (threshold, lower, upper)) or lower >= upper:
        raise ValueError("crossing bounds must be finite and strictly ordered")
    if iterations < 1:
        raise ValueError("crossing iterations must be positive")
    lower_value = float(evaluate_margin(lower)) - threshold
    upper_value = float(evaluate_margin(upper)) - threshold
    if not math.isfinite(lower_value) or not math.isfinite(upper_value):
        raise ValueError("margin evaluator must return finite values")
    if lower_value == 0.0:
        return MarginCrossing(lower, lower, lower)
    if upper_value == 0.0:
        return MarginCrossing(upper, upper, upper)
    if (lower_value > 0.0) == (upper_value > 0.0):
        raise ValueError("crossing bounds do not bracket the threshold")
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_value = float(evaluate_margin(midpoint)) - threshold
        if not math.isfinite(midpoint_value):
            raise ValueError("margin evaluator must return finite values")
        if midpoint_value == 0.0:
            lower = midpoint
            upper = midpoint
            break
        if (midpoint_value > 0.0) == (lower_value > 0.0):
            lower, lower_value = midpoint, midpoint_value
        else:
            upper, upper_value = midpoint, midpoint_value
    return MarginCrossing(lower, upper, (lower + upper) / 2.0)


def random_joint_direction(
    z_shape: tuple[int, ...],
    u_shape: tuple[int, ...],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a CPU Gaussian direction with one joint unit L2 norm."""
    if generator.device.type != "cpu":
        raise ValueError("random_joint_direction requires a CPU generator")
    dz = torch.randn(z_shape, generator=generator)
    du = torch.randn(u_shape, generator=generator)
    norm = torch.linalg.vector_norm(torch.cat((dz.reshape(-1), du.reshape(-1))))
    if norm.item() == 0.0:
        raise RuntimeError("random joint direction has zero norm")
    return dz / norm, du / norm


def select_base_radius(
    acceptance: Mapping[float, Sequence[bool]],
    *,
    minimum_rate: float = 0.8,
    source_acceptance: Mapping[str, Mapping[float, Sequence[bool]]] | None = None,
    minimum_source_rate: float = 0.75,
) -> float | None:
    """Choose the largest radius meeting pooled and optional source-local thresholds."""
    if not 0.0 <= minimum_rate <= 1.0 or not 0.0 <= minimum_source_rate <= 1.0:
        raise ValueError("semantic-acceptance thresholds must be in [0, 1]")
    if source_acceptance is not None and not source_acceptance:
        raise ValueError("source acceptance must include at least one source")
    eligible: list[float] = []
    for radius, outcomes in acceptance.items():
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radii must be finite positive values")
        if not outcomes:
            raise ValueError("each radius needs at least one semantic acceptance value")
        if any(type(outcome) is not bool for outcome in outcomes):
            raise TypeError("semantic acceptance values must be booleans")
        source_passes = True
        if source_acceptance is not None:
            for source, source_by_radius in source_acceptance.items():
                source_outcomes = source_by_radius.get(radius)
                if not source or not source_outcomes:
                    raise ValueError("each source needs acceptance values at every radius")
                if any(type(outcome) is not bool for outcome in source_outcomes):
                    raise TypeError("semantic acceptance values must be booleans")
                if sum(source_outcomes) / len(source_outcomes) < minimum_source_rate:
                    source_passes = False
        if source_passes and sum(outcomes) / len(outcomes) >= minimum_rate:
            eligible.append(float(radius))
    return max(eligible, default=None)


def select_primary_behavioral_radius(
    acceptance: Mapping[float, Sequence[bool]],
    primary_bfr: Mapping[float, float],
    *,
    target_lower: float,
    target_upper: float,
    minimum_semantic_rate: float = 0.8,
) -> float | None:
    """Choose the smallest semantic-valid radius in the primary flip-rate target band."""
    if not 0.0 <= minimum_semantic_rate <= 1.0:
        raise ValueError("minimum semantic rate must be in [0, 1]")
    if not 0.0 <= target_lower <= target_upper <= 1.0:
        raise ValueError("primary BFR target interval must be in [0, 1]")
    if set(acceptance) != set(primary_bfr):
        raise ValueError("semantic acceptance and primary BFR radii must match")
    selected: list[float] = []
    for radius, outcomes in acceptance.items():
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radii must be finite positive values")
        if not outcomes or any(type(outcome) is not bool for outcome in outcomes):
            raise TypeError("semantic acceptance values must be non-empty booleans")
        bfr = primary_bfr[radius]
        if not math.isfinite(bfr) or not 0.0 <= bfr <= 1.0:
            raise ValueError("primary BFR values must be finite probabilities")
        if sum(outcomes) / len(outcomes) >= minimum_semantic_rate and target_lower <= bfr <= target_upper:
            selected.append(float(radius))
    return min(selected, default=None)


def split_fol_bands(values: Sequence[FolValue]) -> FolBandSplit:
    """Select deterministic, disjoint 7/3/7 FOL bands from neutral values."""
    required = LOW_BAND_SIZE + MIDDLE_BAND_SIZE + HIGH_BAND_SIZE
    if len(values) < required:
        raise ValueError(f"need at least {required} FOL values")
    sample_ids = [value.sample_id for value in values]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs are not allowed")
    if any(not math.isfinite(value.fol) for value in values):
        raise ValueError("FOL values must be finite")

    ordered = tuple(sorted(values, key=lambda value: (value.fol, value.sample_id)))
    low_values = ordered[:LOW_BAND_SIZE]
    high_values = ordered[-HIGH_BAND_SIZE:]
    remaining = ordered[LOW_BAND_SIZE:-HIGH_BAND_SIZE]
    center = median(value.fol for value in ordered)
    middle_values = tuple(sorted(
        sorted(remaining, key=lambda value: (abs(value.fol - center), value.fol, value.sample_id))[:MIDDLE_BAND_SIZE],
        key=lambda value: (value.fol, value.sample_id),
    ))
    return FolBandSplit(
        low=tuple(value.sample_id for value in low_values),
        middle=tuple(value.sample_id for value in middle_values),
        high=tuple(value.sample_id for value in high_values),
    )


def behavior_flip_rate(
    accepted_direction_flips: Sequence[bool], *, minimum_accepted: int = 4
) -> BehaviorFlipRate:
    """Compute flips per accepted direction without imputing sparse denominators."""
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    if any(type(flip) is not bool for flip in accepted_direction_flips):
        raise TypeError("accepted direction flips must be booleans")
    accepted_count = len(accepted_direction_flips)
    if accepted_count < minimum_accepted:
        return BehaviorFlipRate(None, accepted_count, True)
    return BehaviorFlipRate(
        rate=sum(accepted_direction_flips) / accepted_count,
        accepted_count=accepted_count,
        acceptance_failure=False,
    )


def exact_permutation_mean_difference(
    low_values: Sequence[float], high_values: Sequence[float]
) -> tuple[float, float]:
    """Return high-minus-low mean difference and exact two-sided label p-value."""
    low = tuple(float(value) for value in low_values)
    high = tuple(float(value) for value in high_values)
    if not low or not high or any(not math.isfinite(value) for value in (*low, *high)):
        raise ValueError("permutation samples must be non-empty and finite")
    observed = sum(high) / len(high) - sum(low) / len(low)
    values = (*low, *high)
    low_count = len(low)
    extreme = 0
    total = 0
    total_sum = sum(values)
    for low_indices in combinations(range(len(values)), low_count):
        low_sum = sum(values[index] for index in low_indices)
        candidate = (total_sum - low_sum) / len(high) - low_sum / len(low)
        if abs(candidate) >= abs(observed) - 1e-12:
            extreme += 1
        total += 1
    return observed, extreme / total


def summarize_fol_bfr(
    rows: Sequence[FolPerturbationOutcome], *, minimum_accepted: int
) -> tuple[FolBfrSummary, ...]:
    """Summarize prompt-level BFR without exposing prompt or response content."""
    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    grouped: dict[tuple[str, str, float], dict[str, list[FolPerturbationOutcome]]] = {}
    for row in rows:
        if not row.source or not row.sample_id or not row.band:
            raise ValueError("FOL outcome identities must be non-empty")
        if not math.isfinite(row.radius) or row.radius <= 0.0:
            raise ValueError("FOL outcome radius must be finite and positive")
        if any(type(value) is not bool for value in (row.accepted, row.primary_label, row.primary_baseline_label)):
            raise TypeError("FOL outcome labels must be booleans")
        if (row.secondary_label is None) is not (row.secondary_baseline_label is None):
            raise ValueError("secondary FOL labels must be both present or both absent")
        if row.secondary_label is not None and any(
            type(value) is not bool for value in (row.secondary_label, row.secondary_baseline_label)
        ):
            raise TypeError("secondary FOL labels must be booleans when present")
        by_sample = grouped.setdefault((row.source, row.band, float(row.radius)), {})
        by_sample.setdefault(row.sample_id, []).append(row)

    summaries: list[FolBfrSummary] = []
    for (source, band, radius), by_sample in sorted(grouped.items()):
        judge_fields = [("primary", "primary_label", "primary_baseline_label")]
        if all(
            row.secondary_label is not None and row.secondary_baseline_label is not None
            for directions in by_sample.values()
            for row in directions
        ):
            judge_fields.append(("secondary", "secondary_label", "secondary_baseline_label"))
        for judge_key, label_field, baseline_field in judge_fields:
            rates: list[float] = []
            accepted_direction_count = 0
            sparse_prompt_count = 0
            for sample_id in sorted(by_sample):
                directions = by_sample[sample_id]
                baselines = {getattr(row, baseline_field) for row in directions}
                if len(baselines) != 1:
                    raise ValueError("FOL outcome baseline labels must be stable per sample")
                baseline = next(iter(baselines))
                flips = [
                    getattr(row, label_field) is not baseline
                    for row in directions
                    if row.accepted
                ]
                rate = behavior_flip_rate(flips, minimum_accepted=minimum_accepted)
                accepted_direction_count += rate.accepted_count
                if rate.acceptance_failure:
                    sparse_prompt_count += 1
                elif rate.rate is not None:
                    rates.append(rate.rate)
            summaries.append(FolBfrSummary(
                source=source,
                band=band,
                radius=radius,
                judge_key=judge_key,
                prompt_count=len(by_sample),
                eligible_prompt_count=len(rates),
                sparse_prompt_count=sparse_prompt_count,
                accepted_direction_count=accepted_direction_count,
                mean_bfr=(sum(rates) / len(rates)) if rates else None,
            ))
    return tuple(summaries)


def interpolate_joint_states(
    safe_z: torch.Tensor,
    safe_u: torch.Tensor,
    unsafe_z: torch.Tensor,
    unsafe_u: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return the fixed 21-point inclusive linear path from safe to unsafe."""
    if safe_z.shape != unsafe_z.shape or safe_u.shape != unsafe_u.shape:
        raise ValueError("safe and unsafe tensors must have matching shapes")
    return tuple(
        (
            torch.lerp(safe_z, unsafe_z, step / (INTERPOLATION_POINT_COUNT - 1)),
            torch.lerp(safe_u, unsafe_u, step / (INTERPOLATION_POINT_COUNT - 1)),
        )
        for step in range(INTERPOLATION_POINT_COUNT)
    )


def validate_interpolation_path(
    semantic_acceptance: Sequence[bool], *, minimum_valid: int = MINIMUM_VALID_INTERPOLATION_POINTS
) -> bool:
    """Require valid endpoints and at least the configured valid 21-point samples."""
    if len(semantic_acceptance) != INTERPOLATION_POINT_COUNT:
        raise ValueError(f"interpolation path must contain exactly {INTERPOLATION_POINT_COUNT} points")
    if not 0 <= minimum_valid <= INTERPOLATION_POINT_COUNT:
        raise ValueError("minimum_valid must fit within the interpolation path")
    if any(type(accepted) is not bool for accepted in semantic_acceptance):
        raise TypeError("semantic acceptance values must be booleans")
    return semantic_acceptance[0] and semantic_acceptance[-1] and sum(semantic_acceptance) >= minimum_valid


def has_minimum_valid_paths(
    valid_paths: Sequence[bool], *, minimum_valid_paths: int = MINIMUM_VALID_INTERPOLATION_PATHS
) -> bool:
    """Apply the pre-registered minimum valid-path threshold."""
    if minimum_valid_paths < 1:
        raise ValueError("minimum_valid_paths must be positive")
    if any(type(path) is not bool for path in valid_paths):
        raise TypeError("valid path values must be booleans")
    return sum(valid_paths) >= minimum_valid_paths


def summarize_interpolation_peaks(
    paths: Sequence[Sequence[InterpolationPoint]],
    *,
    minimum_valid: int,
    minimum_paths: int,
    permutations: int,
    seed: int,
) -> InterpolationPeakSummary:
    """Compare FOL peak proximity to observed crossings without inspecting text.

    A valid path retains the fixed semantic gate.  Its behavior crossing is the
    first adjacent, semantically accepted label transition; paths without a
    transition remain counted for execution quality but do not enter a
    peak-distance comparison.  Random peak locations are sampled independently
    within each eligible path, and the curvature comparison uses a paired,
    one-sided sign test for the pre-registered direction (FOL closer).
    """
    if minimum_paths < 1:
        raise ValueError("minimum_paths must be positive")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    valid_count = 0
    distances: list[tuple[float, float, float]] = []
    for path in paths:
        points = tuple(sorted(path, key=lambda item: item.point_index))
        if len(points) != INTERPOLATION_POINT_COUNT:
            raise ValueError(f"interpolation path must contain exactly {INTERPOLATION_POINT_COUNT} points")
        path_ids = {point.path_id for point in points}
        indices = [point.point_index for point in points]
        if len(path_ids) != 1 or not next(iter(path_ids), ""):
            raise ValueError("interpolation path must have one non-empty path id")
        if indices != list(range(INTERPOLATION_POINT_COUNT)):
            raise ValueError("interpolation point indices must be contiguous")
        if any(type(point.semantic_accepted) is not bool or type(point.unsafe_label) is not bool for point in points):
            raise TypeError("interpolation acceptance and labels must be booleans")
        if any(not math.isfinite(point.fol) or not math.isfinite(point.curvature) for point in points):
            raise ValueError("interpolation FOL and curvature must be finite")
        semantic = [point.semantic_accepted for point in points]
        if not validate_interpolation_path(semantic, minimum_valid=minimum_valid):
            continue
        valid_count += 1
        crossing = next(
            (
                (left.point_index + right.point_index) / 2.0
                for left, right in zip(points, points[1:], strict=True)
                if left.semantic_accepted
                and right.semantic_accepted
                and left.unsafe_label is not right.unsafe_label
            ),
            None,
        )
        if crossing is None:
            continue
        fol_peak = min(
            (point for point in points if point.fol == max(item.fol for item in points)),
            key=lambda point: point.point_index,
        ).point_index
        curvature_peak = min(
            (point for point in points if point.curvature == max(item.curvature for item in points)),
            key=lambda point: point.point_index,
        ).point_index
        distances.append((abs(fol_peak - crossing), abs(curvature_peak - crossing), crossing))

    if valid_count < minimum_paths or len(distances) < minimum_paths:
        return InterpolationPeakSummary(valid_count, len(distances), None, None, None, None, None)
    fol_mean = sum(row[0] for row in distances) / len(distances)
    curvature_mean = sum(row[1] for row in distances) / len(distances)
    generator = random.Random(seed)
    random_means: list[float] = []
    for _ in range(permutations):
        random_means.append(sum(
            abs(generator.randrange(INTERPOLATION_POINT_COUNT) - crossing)
            for _fol_distance, _curvature_distance, crossing in distances
        ) / len(distances))
    random_mean = sum(random_means) / len(random_means)
    random_extreme = sum(value <= fol_mean for value in random_means)
    random_pvalue = (random_extreme + 1) / (len(random_means) + 1)
    non_tied = [curvature - fol for fol, curvature, _crossing in distances if curvature != fol]
    if non_tied:
        positives = sum(value > 0.0 for value in non_tied)
        curvature_pvalue = sum(
            math.comb(len(non_tied), value)
            for value in range(positives, len(non_tied) + 1)
        ) / 2**len(non_tied)
    else:
        curvature_pvalue = None
    return InterpolationPeakSummary(
        valid_count,
        len(distances),
        fol_mean,
        curvature_mean,
        random_mean,
        random_pvalue,
        curvature_pvalue,
    )


def select_nearest_opposite_label_pairs(
    candidates: Sequence[LabeledEditableState],
) -> tuple[OppositeLabelPair, ...]:
    """Choose one safe/unsafe endpoint pair per prompt without using FOL values."""
    by_sample: dict[str, list[LabeledEditableState]] = {}
    for candidate in candidates:
        if not candidate.sample_id or not candidate.candidate_id:
            raise ValueError("editable-state candidate identities must be non-empty")
        if type(candidate.unsafe_label) is not bool:
            raise TypeError("editable-state labels must be booleans")
        vector = candidate.vector.detach()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
            raise ValueError("editable-state vectors must be finite rank-one tensors")
        by_sample.setdefault(candidate.sample_id, []).append(candidate)
    pairs: list[OppositeLabelPair] = []
    for sample_id, rows in sorted(by_sample.items()):
        identifiers = [row.candidate_id for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("editable-state candidate ids must be unique per sample")
        safe = [row for row in rows if not row.unsafe_label]
        unsafe = [row for row in rows if row.unsafe_label]
        if not safe or not unsafe:
            continue
        candidates_with_distance = [
            (
                float(torch.linalg.vector_norm(left.vector.detach().float() - right.vector.detach().float()).item()),
                left.candidate_id,
                right.candidate_id,
            )
            for left in safe
            for right in unsafe
        ]
        distance, safe_id, unsafe_id = min(candidates_with_distance)
        pairs.append(OppositeLabelPair(sample_id, safe_id, unsafe_id, distance))
    return tuple(pairs)
