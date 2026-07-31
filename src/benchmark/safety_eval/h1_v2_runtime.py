"""Shared, content-free eligibility checks for the H1-v2 confirmation run."""

from __future__ import annotations

import json
from pathlib import Path

from .fol_records import resolved_terminal_payloads
from .fol_runtime import select_h1_v2_eligible_ids
from .io import read_jsonl
from .schema import BenchmarkExample, OptimizationRecord, RecordStatus


_EXCLUSION_ARTIFACT = "h1_v2_computational_exclusions.json"


def h1_v2_computational_exclusions(root: Path, source: str) -> tuple[str, ...]:
    """Load the single, audited computational exclusion for a source, if present."""
    try:
        payload = json.loads((root / "manifests" / _EXCLUSION_ARTIFACT).read_text(encoding="utf-8"))
        run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("H1-v2 computational exclusion artifact is unreadable") from error
    if payload.get("config_hash") != run.get("config_hash"):
        raise ValueError("H1-v2 computational exclusion configuration hash differs from the run")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("H1-v2 computational exclusion artifact has no source map")
    values = sources.get(source)
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ValueError("H1-v2 computational exclusion source list is invalid")
    return tuple(sorted(values))


def h1_v2_eligible_records(root: Path, source: str) -> tuple[dict[str, BenchmarkExample], dict[str, OptimizationRecord]]:
    """Return only manifest candidates with resolved terminal O+ states."""
    examples = {
        row.example_id: row
        for row in (
            BenchmarkExample.model_validate(item)
            for item in read_jsonl(root / "manifests" / f"controlled_{source}.jsonl")
        )
    }
    optimized = {
        row.sample_id: row
        for row in (
            OptimizationRecord.model_validate(item)
            for item in resolved_terminal_payloads(root, source).values()
        )
        if row.checkpoint == 100 and row.status is RecordStatus.complete and row.state_path
    }
    eligible_ids = select_h1_v2_eligible_ids(
        manifest_ids=tuple(examples), terminal_ids=tuple(optimized),
        computational_exclusions=h1_v2_computational_exclusions(root, source),
    )
    return (
        {sample_id: examples[sample_id] for sample_id in eligible_ids},
        {sample_id: optimized[sample_id] for sample_id in eligible_ids},
    )
