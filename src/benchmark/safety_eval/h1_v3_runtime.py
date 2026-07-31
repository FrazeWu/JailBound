"""Content-free runtime helpers for the isolated H1-v3 radius extension."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .fol_runtime import PerturbationScheduleRow
from .h1_v2_runtime import h1_v2_eligible_records
from .io import canonical_hash


_SOURCES = ("jailbound", "s_eval")
_BANDS = (("low", 17), ("middle", 3), ("high", 17))
_RADII = (0.4, 0.6)


@dataclass(frozen=True)
class H1V3Paths:
    """Resolve H1-v3 writes beneath a distinct root from frozen H1-v2 inputs."""

    source_root: Path
    output_root: Path

    def __post_init__(self) -> None:
        if self.source_root.name != "fol_h1_v2" or self.output_root.name != "fol_h1_v3":
            raise ValueError("H1-v3 paths require fol_h1_v2 source and fol_h1_v3 output roots")
        if self.source_root == self.output_root:
            raise ValueError("H1-v3 source and output roots must differ")

    def source_selection(self) -> Path:
        return self.source_root / "manifests" / "h1_v2_validation_selection.json"

    def source_run_manifest(self) -> Path:
        return self.source_root / "run_manifest.json"

    def schedule(self) -> Path:
        return self.output_root / "direction_schedule.jsonl"

    def schedule_metadata(self) -> Path:
        return self.output_root / "direction_schedule.json"

    def semantic_ledger(self) -> Path:
        return self.output_root / "direction_semantic_outcomes.jsonl"

    def selected_directions(self) -> Path:
        return self.output_root / "selected_directions.jsonl"

    def response_ledger(self, target_key: str, source: str) -> Path:
        return self.output_root / "responses" / target_key / source / "fol_h1_v3" / "records.jsonl"

    def judgment_ledger(self, target_key: str, source: str) -> Path:
        return self.output_root / "judgments" / "octopus_seval_14b" / target_key / source / "fol_h1_v3" / "records.jsonl"

    def direction_labels(self) -> Path:
        return self.output_root / "direction_labels.jsonl"


def frozen_h1_v2_selection(source_root: Path) -> dict[str, dict[str, Any]]:
    """Read and validate frozen opaque H1-v2 selection identities only."""
    paths = H1V3Paths(source_root=source_root, output_root=source_root.parent / "fol_h1_v3")
    try:
        payload = json.loads(paths.source_selection().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("H1-v3 frozen H1-v2 selection is unreadable") from error
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, dict):
        raise ValueError("H1-v3 frozen H1-v2 selection has no source map")
    validated: dict[str, dict[str, Any]] = {}
    for source in _SOURCES:
        bands = sources.get(source)
        if not isinstance(bands, dict):
            raise ValueError(f"H1-v3 frozen selection lacks {source}")
        normalized: dict[str, Any] = {}
        identities: list[str] = []
        for band, expected_count in _BANDS:
            values = bands.get(band)
            if not isinstance(values, list) or len(values) != expected_count:
                raise ValueError(f"H1-v3 frozen selection has invalid {source}/{band} IDs")
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError(f"H1-v3 frozen selection has non-opaque {source}/{band} IDs")
            normalized[band] = tuple(values)
            identities.extend(values)
        if len(set(identities)) != 37:
            raise ValueError(f"H1-v3 frozen selection reuses {source} identities")
        fol_by_id = bands.get("fol_by_id")
        if not isinstance(fol_by_id, dict) or not set(identities) <= set(fol_by_id):
            raise ValueError(f"H1-v3 frozen selection has invalid {source} FOL values")
        try:
            normalized["fol_by_id"] = {sample_id: float(fol_by_id[sample_id]) for sample_id in identities}
        except (TypeError, ValueError) as error:
            raise ValueError(f"H1-v3 frozen selection has nonnumeric {source} FOL values") from error
        if not all(math.isfinite(value) for value in normalized["fol_by_id"].values()):
            raise ValueError(f"H1-v3 frozen selection has nonfinite {source} FOL values")
        validated[source] = normalized
    return validated


def frozen_h1_v2_records(source_root: Path, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return frozen selected examples and O+ states from H1-v2, never H1-v3."""
    if source not in _SOURCES:
        raise ValueError("H1-v3 source is not configured")
    selection = frozen_h1_v2_selection(source_root)[source]
    selected = {sample_id for band, _ in _BANDS for sample_id in selection[band]}
    examples, optimized = h1_v2_eligible_records(source_root, source)
    if not selected <= set(examples) or not selected <= set(optimized):
        raise ValueError("H1-v3 frozen H1-v2 selected records are unavailable")
    return (
        {sample_id: examples[sample_id] for sample_id in sorted(selected)},
        {sample_id: optimized[sample_id] for sample_id in sorted(selected)},
    )


def frozen_h1_v2_selected_ids(source_root: Path) -> tuple[tuple[str, str], ...]:
    """Return source-qualified opaque identities in a stable order."""
    selection = frozen_h1_v2_selection(source_root)
    return tuple(
        sorted(
            (source, sample_id)
            for source in _SOURCES
            for band, _ in _BANDS
            for sample_id in selection[source][band]
        )
    )


def build_h1_v3_schedule(
    *, sample_ids: Sequence[str], radii: Sequence[float], attempts: int, seed: int
) -> tuple[PerturbationScheduleRow, ...]:
    """Freeze new-radius H1-v3 direction identities before semantic evaluation."""
    if attempts < 1:
        raise ValueError("H1-v3 maximum direction attempts must be positive")
    ids = tuple(sorted(sample_ids))
    if not ids or len(ids) != len(set(ids)) or any(not sample_id for sample_id in ids):
        raise ValueError("H1-v3 schedule needs unique non-empty sample IDs")
    ordered_radii = tuple(sorted(float(radius) for radius in radii))
    if ordered_radii != _RADII:
        raise ValueError("H1-v3 schedule requires exactly [0.4, 0.6]")
    rows = []
    for sample_id in ids:
        for radius in ordered_radii:
            for direction_index in range(attempts):
                digest = hashlib.sha256(
                    f"{seed}|h1-v3|{sample_id}|{radius:.17g}|{direction_index}".encode()
                ).hexdigest()
                rows.append(PerturbationScheduleRow(
                    perturbation_id=f"fol-h1-v3:{digest[:20]}",
                    sample_id=sample_id,
                    radius=radius,
                    direction_index=direction_index,
                    direction_seed=int(digest[:16], 16) % (2**31),
                ))
    return tuple(rows)


def build_h1_v3_contract(
    *, source_hash: str, selected_ids: Sequence[tuple[str, str]], radii: Sequence[float]
) -> dict[str, object]:
    """Create a numeric-only contract binding H1-v3 to frozen H1-v2 inputs."""
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("H1-v3 contract requires a SHA-256 source configuration hash")
    identities = tuple(sorted(selected_ids))
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("H1-v3 contract requires unique selected prompt identities")
    if tuple(float(radius) for radius in radii) != _RADII:
        raise ValueError("H1-v3 contract requires exactly [0.4, 0.6]")
    return {
        "analysis_status": "follow_up_extension",
        "source_h1_v2_config_hash": source_hash,
        "selected_ids_hash": canonical_hash(identities),
        "new_radii": list(_RADII),
        "accepted_directions": 32,
        "primary_judge": "octopus_seval_14b",
        "target_model": "qwen2_5_7b",
    }


def validate_h1_v3_contract(
    contract: dict[str, object], *, source_hash: str, selected_ids: Sequence[tuple[str, str]], radii: Sequence[float]
) -> None:
    """Fail closed if any frozen source identity or new-radius value changes."""
    if contract != build_h1_v3_contract(
        source_hash=source_hash, selected_ids=selected_ids, radii=radii
    ):
        raise ValueError("H1-v3 contract does not match frozen inputs")
