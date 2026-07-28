"""Content-free selection contracts for the focused FOL boundary experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import torch

from .manifest import FolCandidate, select_controlled, select_fol_validation
from .schema import BenchmarkExample


FOL_CANDIDATE_COUNT = 45
FOL_VALIDATION_COUNT = 17
FOL_RADIUS_CALIBRATION_COUNT = 5
FOL_LOW_COUNT = 7
FOL_MIDDLE_COUNT = 3
FOL_HIGH_COUNT = 7


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
class PerturbationScheduleRow:
    """A pre-registered random-direction identity with no model-derived fields."""

    perturbation_id: str
    sample_id: str
    radius: float
    direction_index: int
    direction_seed: int


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
