"""Content-free selection contracts for the focused FOL boundary experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch

from .manifest import FolCandidate, select_controlled, select_fol_validation
from .schema import BenchmarkExample


FOL_CANDIDATE_COUNT = 45
FOL_VALIDATION_COUNT = 17
FOL_RADIUS_CALIBRATION_COUNT = 5
FOL_LOW_COUNT = 7
FOL_MIDDLE_COUNT = 3
FOL_HIGH_COUNT = 7
H1_V2_CANDIDATE_COUNT = 81
H1_V2_MINIMUM_BASELINE_SAFE_COUNT = 41
H1_V2_LOW_COUNT = 17
H1_V2_MIDDLE_COUNT = 3
H1_V2_HIGH_COUNT = 17
H1_V2_RESERVE_COUNT = 4


@dataclass(frozen=True)
class FolExperimentSelection:
    """Stable, non-overlapping source-local FOL experiment identities."""

    low: tuple[str, ...]
    middle: tuple[str, ...]
    high: tuple[str, ...]
    radius_calibration: tuple[str, ...]
    status: str
    unmatched: tuple[str, ...] = ()
    matching_caliper: float | None = None
    matching_distances: tuple[float, ...] = ()


@dataclass(frozen=True)
class ConfirmatoryFolCandidate:
    """A numeric, baseline-safe candidate eligible for independent H1-v2 selection."""

    sample_id: str
    source: str
    fol: float
    risk_category: str
    attack_loss: float
    token_length: int
    perplexity: float
    baseline_safe: bool


@dataclass(frozen=True)
class ConfirmatoryFolSelection:
    """Frozen H1-v2 endpoints and ordered replacements for one source."""

    low: tuple[str, ...]
    middle: tuple[str, ...]
    high: tuple[str, ...]
    reserves: tuple[str, ...]
    status: str
    unmatched: tuple[str, ...] = ()
    matching_caliper: float | None = None
    matching_distances: tuple[float, ...] = ()
    risk_category_matching: bool = True
    matching_mode: str = "risk_category_matched"


@dataclass(frozen=True)
class PerturbationScheduleRow:
    """A pre-registered random-direction identity with no model-derived fields."""

    perturbation_id: str
    sample_id: str
    radius: float
    direction_index: int
    direction_seed: int


@dataclass(frozen=True)
class H1V2DirectionSelection:
    """Accepted frozen direction identities and explicit incomplete groups."""

    accepted: tuple[PerturbationScheduleRow, ...]
    insufficient: tuple[tuple[str, float, int], ...]


def select_id_shard(
    identifiers: Sequence[str], *, shard_index: int, shard_count: int
) -> tuple[str, ...]:
    """Return one deterministic, disjoint shard of unique identifiers."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be within a positive shard count")
    ordered = tuple(sorted(identifiers))
    if any(not identifier for identifier in ordered) or len(ordered) != len(set(ordered)):
        raise ValueError("identifiers must be unique and non-empty")
    return tuple(identifier for index, identifier in enumerate(ordered) if index % shard_count == shard_index)


def select_schedule_shard(
    schedule: Sequence[PerturbationScheduleRow], *, shard_index: int, shard_count: int
) -> tuple[PerturbationScheduleRow, ...]:
    """Return one deterministic, disjoint shard of frozen perturbation identities."""
    ordered = tuple(sorted(schedule, key=lambda row: row.perturbation_id))
    identities = [row.perturbation_id for row in ordered]
    selected = set(select_id_shard(identities, shard_index=shard_index, shard_count=shard_count))
    return tuple(row for row in ordered if row.perturbation_id in selected)


def causal_perplexity(logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    """Return next-token perplexity from one causal-LM forward pass."""
    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("logits must be [batch, tokens, vocabulary] and token IDs [batch, tokens]")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token IDs must share batch and token dimensions")
    if token_ids.shape[1] < 2:
        raise ValueError("causal perplexity requires at least two tokens")
    if token_ids.dtype.is_floating_point or token_ids.dtype.is_complex:
        raise ValueError("token IDs must be integral")
    targets = token_ids[:, 1:].to(device=logits.device, dtype=torch.long)
    if targets.numel() == 0 or targets.min().item() < 0 or targets.max().item() >= logits.shape[-1]:
        raise ValueError("token IDs are outside the logits vocabulary")
    log_probabilities = torch.log_softmax(logits[:, :-1, :], dim=-1)
    negative_log_likelihood = -log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
    value = float(torch.exp(negative_log_likelihood).detach().cpu())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("causal perplexity must be finite and positive")
    return value


def build_perturbation_schedule(
    *,
    sample_ids: Sequence[str],
    radii: Sequence[float],
    directions_per_radius: int,
    seed: int,
) -> tuple[PerturbationScheduleRow, ...]:
    """Freeze independent random-direction identities before model evaluation."""
    if directions_per_radius < 1:
        raise ValueError("directions_per_radius must be positive")
    ids = tuple(sorted(sample_ids))
    if not ids or any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("perturbation schedule needs unique non-empty sample IDs")
    ordered_radii = tuple(sorted(float(radius) for radius in radii))
    if not ordered_radii or len(ordered_radii) != len(set(ordered_radii)):
        raise ValueError("perturbation schedule needs unique radii")
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in ordered_radii):
        raise ValueError("perturbation schedule radii must be finite positive values")
    schedule = []
    for sample_id in ids:
        for radius in ordered_radii:
            for direction_index in range(directions_per_radius):
                payload = f"{seed}|{sample_id}|{radius:.17g}|{direction_index}"
                digest = hashlib.sha256(payload.encode()).hexdigest()
                schedule.append(PerturbationScheduleRow(
                    perturbation_id=f"fol:{digest[:20]}",
                    sample_id=sample_id,
                    radius=radius,
                    direction_index=direction_index,
                    direction_seed=int(digest[:16], 16) % (2**31),
                ))
    return tuple(schedule)


def select_accepted_perturbations(
    schedule: Sequence[PerturbationScheduleRow],
    *,
    accepted_by_id: dict[str, bool],
    directions_per_radius: int,
) -> tuple[PerturbationScheduleRow, ...]:
    """Keep the earliest accepted frozen directions up to each group capacity."""
    if directions_per_radius < 1:
        raise ValueError("directions_per_radius must be positive")
    schedule_ids = [row.perturbation_id for row in schedule]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("perturbation schedule IDs must be unique")
    if set(accepted_by_id) != set(schedule_ids):
        raise ValueError("semantic acceptance IDs must exactly match the perturbation schedule")
    if any(type(accepted) is not bool for accepted in accepted_by_id.values()):
        raise TypeError("semantic acceptance values must be booleans")
    selected: list[PerturbationScheduleRow] = []
    counts: dict[tuple[str, float], int] = {}
    for row in sorted(schedule, key=lambda value: (value.sample_id, value.radius, value.direction_index)):
        group = (row.sample_id, row.radius)
        if accepted_by_id[row.perturbation_id] and counts.get(group, 0) < directions_per_radius:
            selected.append(row)
            counts[group] = counts.get(group, 0) + 1
    return tuple(selected)


def build_h1_v2_direction_schedule(
    *, sample_ids: Sequence[str], radii: Sequence[float], max_direction_attempts: int, seed: int
) -> tuple[PerturbationScheduleRow, ...]:
    """Freeze the oversampled H1-v2 schedule before semantic checks run."""
    if max_direction_attempts < 1:
        raise ValueError("H1-v2 max direction attempts must be positive")
    ids = tuple(sorted(sample_ids))
    if not ids or any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("H1-v2 schedule needs unique non-empty sample IDs")
    ordered_radii = tuple(sorted(float(radius) for radius in radii))
    if not ordered_radii or len(ordered_radii) != len(set(ordered_radii)):
        raise ValueError("H1-v2 schedule needs unique radii")
    if any(not math.isfinite(radius) or radius <= 0.0 for radius in ordered_radii):
        raise ValueError("H1-v2 schedule radii must be finite positive values")
    rows = []
    for sample_id in ids:
        for radius in ordered_radii:
            for direction_index in range(max_direction_attempts):
                payload = f"{seed}|h1-v2|{sample_id}|{radius:.17g}|{direction_index}"
                digest = hashlib.sha256(payload.encode()).hexdigest()
                rows.append(PerturbationScheduleRow(
                    perturbation_id=f"fol-h1-v2:{digest[:20]}",
                    sample_id=sample_id,
                    radius=radius,
                    direction_index=direction_index,
                    direction_seed=int(digest[:16], 16) % (2**31),
                ))
    return tuple(rows)


def select_first_accepted_directions(
    schedule: Sequence[PerturbationScheduleRow], *, accepted_by_id: dict[str, bool], required_count: int
) -> H1V2DirectionSelection:
    """Select the first accepted directions without replacement or reordering."""
    if required_count < 1:
        raise ValueError("H1-v2 required accepted direction count must be positive")
    schedule_ids = [row.perturbation_id for row in schedule]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("H1-v2 direction schedule IDs must be unique")
    if set(accepted_by_id) != set(schedule_ids):
        raise ValueError("H1-v2 semantic acceptance IDs must exactly match the schedule")
    if any(type(value) is not bool for value in accepted_by_id.values()):
        raise TypeError("H1-v2 semantic acceptance values must be booleans")
    groups: dict[tuple[str, float], list[PerturbationScheduleRow]] = {}
    for row in sorted(schedule, key=lambda value: (value.sample_id, value.radius, value.direction_index)):
        if accepted_by_id[row.perturbation_id]:
            groups.setdefault((row.sample_id, row.radius), []).append(row)
        else:
            groups.setdefault((row.sample_id, row.radius), [])
    accepted: list[PerturbationScheduleRow] = []
    insufficient: list[tuple[str, float, int]] = []
    for (sample_id, radius), rows in sorted(groups.items()):
        if len(rows) < required_count:
            insufficient.append((sample_id, radius, len(rows)))
            continue
        accepted.extend(rows[:required_count])
    return H1V2DirectionSelection(tuple(accepted), tuple(insufficient))


def select_fol_candidates(
    rows: Sequence[BenchmarkExample], *, excluded_ids: set[str] | frozenset[str], seed: int
) -> tuple[BenchmarkExample, ...]:
    """Freeze 45 source-local FOL candidates disjoint from the main matrix."""
    eligible = tuple(row for row in rows if row.example_id not in excluded_ids)
    if not eligible:
        raise ValueError("FOL candidate selection has no rows after main-matrix exclusion")
    sources = {row.source for row in eligible}
    if len(sources) != 1:
        raise ValueError("FOL candidate selection requires exactly one source")
    source = next(iter(sources))
    dimensions = (
        ("risk_category", "threat_domain", "attack_type")
        if source == "jailbound"
        else ("risk_category", "attack_type")
    )
    selected, _ = select_controlled(
        eligible,
        n=FOL_CANDIDATE_COUNT,
        seed=seed,
        coverage_dimensions=dimensions,
    )
    return selected


def select_h1_v2_candidates(
    rows: Sequence[BenchmarkExample], *, excluded_ids: set[str] | frozenset[str], seed: int
) -> tuple[BenchmarkExample, ...]:
    """Freeze 81 new candidates, disjoint from all prior FOL identities."""
    eligible = tuple(row for row in rows if row.example_id not in excluded_ids)
    if len(eligible) < H1_V2_CANDIDATE_COUNT:
        raise ValueError("H1-v2 has fewer than 81 candidates after identity exclusion")
    sources = {row.source for row in eligible}
    if len(sources) != 1:
        raise ValueError("H1-v2 candidate selection requires exactly one source")
    source = next(iter(sources))
    dimensions = (
        ("risk_category", "threat_domain", "attack_type")
        if source == "jailbound"
        else ("risk_category", "attack_type")
    )
    selected, _ = select_controlled(
        eligible,
        n=H1_V2_CANDIDATE_COUNT,
        seed=seed,
        coverage_dimensions=dimensions,
    )
    selected_ids = {row.example_id for row in selected}
    if selected_ids & excluded_ids:
        raise AssertionError("H1-v2 candidate identity exclusion was violated")
    return selected


def select_fol_experiment(
    rows: Sequence[FolCandidate], *, seed: int
) -> FolExperimentSelection:
    """Select registered bands and five disjoint radius-calibration IDs."""
    if len(rows) < FOL_CANDIDATE_COUNT:
        raise ValueError(f"FOL experiment requires at least {FOL_CANDIDATE_COUNT} candidates")
    candidate_ids = [row.sample_id for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("FOL experiment candidates must have unique sample IDs")
    split = select_fol_validation(
        rows,
        validation_n=FOL_VALIDATION_COUNT,
        low_n=FOL_LOW_COUNT,
        middle_n=FOL_MIDDLE_COUNT,
        high_n=FOL_HIGH_COUNT,
    )
    if split.status != "ready":
        return FolExperimentSelection((), (), (), (), split.status, split.unmatched)
    low = tuple(row.sample_id for row in split.low)
    middle = tuple(row.sample_id for row in split.middle)
    high = tuple(row.sample_id for row in split.high)
    validation_ids = frozenset((*low, *middle, *high))
    remaining = [row.sample_id for row in rows if row.sample_id not in validation_ids]
    ranked = sorted(
        remaining,
        key=lambda sample_id: hashlib.sha256(f"{seed}|radius-calibration|{sample_id}".encode()).hexdigest(),
    )
    calibration = tuple(ranked[:FOL_RADIUS_CALIBRATION_COUNT])
    if len(calibration) != FOL_RADIUS_CALIBRATION_COUNT:
        raise ValueError("FOL experiment lacks disjoint radius-calibration candidates")
    return FolExperimentSelection(
        low, middle, high, calibration, "ready",
        matching_caliper=split.matching_caliper,
        matching_distances=split.matching_distances,
    )


def select_h1_v2_validation(
    rows: Sequence[ConfirmatoryFolCandidate], *, exploratory_ids: set[str] | frozenset[str], seed: int
) -> ConfirmatoryFolSelection:
    """Lock baseline-safe 17/3/17 bands and four deterministic replacement IDs.

    The caller supplies only numeric diagnostics and the primary baseline label.
    This prevents selection from accessing prompts, model responses, or perturbed
    behavioral labels.
    """
    candidate_ids = tuple(row.sample_id for row in rows)
    if len(candidate_ids) != len(set(candidate_ids)) or any(not sample_id for sample_id in candidate_ids):
        raise ValueError("H1-v2 candidates must have unique non-empty sample IDs")
    sources = {row.source for row in rows}
    if len(sources) != 1:
        raise ValueError("H1-v2 validation selection requires exactly one source")
    if set(candidate_ids) & set(exploratory_ids):
        raise ValueError("H1-v2 validation candidates overlap exploratory identities")
    baseline_safe = tuple(row for row in rows if row.baseline_safe)
    if len(baseline_safe) < H1_V2_MINIMUM_BASELINE_SAFE_COUNT:
        return ConfirmatoryFolSelection(
            (), (), (), (), "inconclusive", tuple(sorted(row.sample_id for row in rows if not row.baseline_safe))
        )
    if len(baseline_safe) != len(rows):
        raise ValueError("H1-v2 selection must receive the frozen baseline-safe stratum only")
    compatible = tuple(
        FolCandidate(
            sample_id=row.sample_id,
            source=row.source,
            fol=row.fol,
            risk_category=row.risk_category,
            initial_label=False,
            attack_loss=row.attack_loss,
            token_length=row.token_length,
            perplexity=row.perplexity,
        )
        for row in baseline_safe
    )
    split = select_fol_validation(
        compatible,
        validation_n=H1_V2_LOW_COUNT + H1_V2_MIDDLE_COUNT + H1_V2_HIGH_COUNT,
        low_n=H1_V2_LOW_COUNT,
        middle_n=H1_V2_MIDDLE_COUNT,
        high_n=H1_V2_HIGH_COUNT,
    )
    risk_category_matching = True
    matching_mode = "risk_category_matched"
    if split.status != "ready":
        split = select_fol_validation(
            compatible,
            validation_n=H1_V2_LOW_COUNT + H1_V2_MIDDLE_COUNT + H1_V2_HIGH_COUNT,
            low_n=H1_V2_LOW_COUNT,
            middle_n=H1_V2_MIDDLE_COUNT,
            high_n=H1_V2_HIGH_COUNT,
            match_risk_category=False,
        )
        risk_category_matching = False
        matching_mode = "covariate_matched_without_risk"
    if split.status != "ready":
        # The pre-registered final fallback retains endpoint separation when no
        # covariate-matched design exists. Its unmatched status is explicit in
        # the artifact and downstream reporting.
        ordered = tuple(sorted(baseline_safe, key=lambda row: (row.fol, row.sample_id)))
        low = tuple(row.sample_id for row in ordered[:H1_V2_LOW_COUNT])
        high = tuple(row.sample_id for row in ordered[-H1_V2_HIGH_COUNT:])
        used = set((*low, *high))
        median = math.fsum(row.fol for row in ordered) / len(ordered)
        middle = tuple(
            row.sample_id
            for row in sorted(
                (row for row in ordered if row.sample_id not in used),
                key=lambda row: (abs(row.fol - median), row.sample_id),
            )[:H1_V2_MIDDLE_COUNT]
        )
        matching_caliper = None
        matching_distances: tuple[float, ...] = ()
        risk_category_matching = False
        matching_mode = "unmatched_endpoints"
    else:
        low = tuple(row.sample_id for row in split.low)
        middle = tuple(row.sample_id for row in split.middle)
        high = tuple(row.sample_id for row in split.high)
        matching_caliper = split.matching_caliper
        matching_distances = split.matching_distances
    selected = set((*low, *middle, *high))
    remaining = [row.sample_id for row in baseline_safe if row.sample_id not in selected]
    reserves = tuple(sorted(
        remaining,
        key=lambda sample_id: hashlib.sha256(f"{seed}|h1-v2-reserve|{sample_id}".encode()).hexdigest(),
    )[:H1_V2_RESERVE_COUNT])
    if len(reserves) != H1_V2_RESERVE_COUNT:
        return ConfirmatoryFolSelection((), (), (), (), "inconclusive", tuple(sorted(remaining)))
    all_ids = (*low, *middle, *high, *reserves)
    if len(set(all_ids)) != len(all_ids):
        raise AssertionError("H1-v2 selected or reserve IDs overlap")
    return ConfirmatoryFolSelection(
        low, middle, high, reserves, "ready",
        matching_caliper=matching_caliper,
        matching_distances=matching_distances,
        risk_category_matching=risk_category_matching,
        matching_mode=matching_mode,
    )


def select_h1_v2_oom_recovery_ids(records: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Return only candidates whose registered primary checkpoints all OOMed."""
    expected = {0, 25, 50, 100}
    by_sample: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str) and sample_id:
            by_sample.setdefault(sample_id, []).append(record)
    recoverable = []
    for sample_id, sample_records in by_sample.items():
        by_checkpoint = {record.get("checkpoint"): record for record in sample_records}
        if set(by_checkpoint) != expected:
            continue
        if all(
            record.get("status") == "failed"
            and record.get("failure_kind") == "optimization"
            and isinstance(record.get("failure_reason"), str)
            and "OutOfMemoryError" in str(record["failure_reason"])
            for record in by_checkpoint.values()
        ):
            recoverable.append(sample_id)
    return tuple(sorted(recoverable))


def select_h1_v2_pending_recovery_ids(
    primary_records: Sequence[Mapping[str, object]], recovery_records: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    """Exclude primary OOM candidates that already have a terminal successful recovery."""
    recovered = {
        str(record["sample_id"])
        for record in recovery_records
        if record.get("checkpoint") == 100 and record.get("status") == "complete"
        and isinstance(record.get("sample_id"), str) and record.get("sample_id")
    }
    return tuple(sample_id for sample_id in select_h1_v2_oom_recovery_ids(primary_records) if sample_id not in recovered)


def select_h1_v2_eligible_ids(
    *, manifest_ids: Sequence[str], terminal_ids: Sequence[str], computational_exclusions: Sequence[str]
) -> tuple[str, ...]:
    """Return the frozen candidates with only documented unresolved OOM exclusions removed."""
    manifest = tuple(sorted(manifest_ids))
    terminal = frozenset(terminal_ids)
    exclusions = frozenset(computational_exclusions)
    if len(manifest) != H1_V2_CANDIDATE_COUNT or len(manifest) != len(set(manifest)) or any(not value for value in manifest):
        raise ValueError("H1-v2 manifest candidates are invalid")
    if not terminal <= set(manifest):
        raise ValueError("H1-v2 terminal diagnostics reference an unknown candidate")
    if not exclusions < set(manifest):
        raise ValueError("H1-v2 computational exclusions must identify manifest candidates")
    unresolved = set(manifest) - terminal
    if unresolved != set(exclusions):
        raise ValueError("H1-v2 unresolved candidates do not exactly match computational exclusions")
    return tuple(sample_id for sample_id in manifest if sample_id not in exclusions)


def h1_v2_recovery_spec(mode: str) -> tuple[str, bool]:
    """Return the isolated recovery method and its activation-checkpoint flag."""
    if mode == "eager":
        return "jailbound_o_plus_recovery_eager", False
    if mode == "checkpointed":
        return "jailbound_o_plus_recovery_checkpointed", True
    if mode == "sdpa":
        return "jailbound_o_plus_recovery_sdpa", False
    if mode == "cpu_offload":
        return "jailbound_o_plus_recovery_cpu_offload", False
    if mode == "two_gpu":
        return "jailbound_o_plus_recovery_two_gpu", False
    if mode == "two_gpu_checkpointed":
        return "jailbound_o_plus_recovery_two_gpu_checkpointed", True
    raise ValueError(
        "H1-v2 recovery mode must be eager, checkpointed, sdpa, cpu_offload, two_gpu, or two_gpu_checkpointed"
    )
