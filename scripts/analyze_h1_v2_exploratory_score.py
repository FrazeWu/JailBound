"""Materialize a post-hoc continuous-score companion analysis for H1-v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.h1_v2_analysis import (
    PromptScoreCurve,
    score_auc,
    source_stratified_score_association,
)
from benchmark.safety_eval.io import atomic_write_json, canonical_hash, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
STATUS = "post_hoc_exploratory"
ENDPOINT = "mean_unsafe_score_auc_log_radius"
TEST = "source_stratified_spearman_permutation_two_sided"


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"exploratory score {label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"exploratory score {label} must be an object")
    return payload


def _selection_ids(selection: dict[str, Any], sources: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    values = selection.get("sources")
    if not isinstance(values, dict):
        raise ValueError("exploratory score selection has no source map")
    selected: list[tuple[str, str]] = []
    for source in sources:
        bands = values.get(source)
        if not isinstance(bands, dict):
            raise ValueError(f"exploratory score selection lacks {source}")
        for band, expected in (("low", 17), ("middle", 3), ("high", 17)):
            ids = bands.get(band)
            if not isinstance(ids, list) or len(ids) != expected or not all(isinstance(item, str) and item for item in ids):
                raise ValueError(f"exploratory score selection has invalid {source}/{band} IDs")
            selected.extend((source, item) for item in ids)
    if len(selected) != len(set(selected)):
        raise ValueError("exploratory score selection reuses a prompt identity")
    return tuple(sorted(selected))


def _radii(calibration: dict[str, Any]) -> tuple[float, ...]:
    raw = calibration.get("r_local")
    if not isinstance(raw, list):
        raise ValueError("exploratory score calibration has no local radii")
    values = tuple(float(value) for value in raw)
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in values) or any(
        later <= earlier for earlier, later in zip(values, values[1:], strict=False)
    ):
        raise ValueError("exploratory score radii must be strictly increasing and positive")
    return values


def _contract(*, config_hash: str, selected_ids: tuple[tuple[str, str], ...], radii: tuple[float, ...]) -> dict[str, object]:
    return {
        "analysis_status": STATUS,
        "inference_scope": "hypothesis_generating_only",
        "config_hash": config_hash,
        "selected_ids_hash": canonical_hash(tuple(sorted(selected_ids))),
        "local_radii": list(radii),
        "accepted_directions": 32,
        "endpoint": ENDPOINT,
        "test": TEST,
        "judge_field": "unsafe_score",
        "disclaimer": "This post-hoc exploratory analysis does not replace the preregistered H1-v2 binary UTR-AUC endpoint.",
    }


def _curves(
    *, root: Path, config: Any, selection: dict[str, Any], selected_ids: tuple[tuple[str, str], ...], radii: tuple[float, ...]
) -> tuple[PromptScoreCurve, ...]:
    source_map = selection["sources"]
    fol = {
        (source, sample_id): float(source_map[source]["fol_by_id"][sample_id])
        for source, sample_id in selected_ids
    }
    baselines = {
        (str(row.get("source")), str(row.get("sample_id"))): row
        for row in read_jsonl(root / "baseline_labels.jsonl")
    }
    directions = read_jsonl(root / "selected_directions.jsonl")
    target_key = config.base.models.targets[0].key
    threshold = config.base.judging.primary.threshold
    scores: dict[str, float] = {}
    for source in config.h1_v2.sources:
        path = root / "judgments" / "octopus_seval_14b" / target_key / source / "fol_h1_v2" / "records.jsonl"
        for row in read_jsonl(path):
            if row.get("checkpoint") != 0 or row.get("threshold") != threshold or row.get("status") != "complete":
                continue
            sample_id = row.get("sample_id")
            score = row.get("unsafe_score")
            if isinstance(sample_id, str) and isinstance(score, (int, float)) and not isinstance(score, bool):
                scores[sample_id] = float(score)
    grouped: dict[tuple[str, str, float], list[float]] = {}
    for row in directions:
        source, sample_id, perturbation_id, radius = (
            row.get("source"), row.get("sample_id"), row.get("perturbation_id"), row.get("radius")
        )
        if not isinstance(source, str) or not isinstance(sample_id, str) or not isinstance(perturbation_id, str):
            raise ValueError("exploratory score direction identity is invalid")
        if (source, sample_id) not in fol:
            continue
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or perturbation_id not in scores:
            raise ValueError("exploratory score directions are incomplete")
        grouped.setdefault((source, sample_id, float(radius)), []).append(scores[perturbation_id])
    curves = []
    for source, sample_id in selected_ids:
        baseline = baselines.get((source, sample_id))
        if baseline is None or baseline.get("baseline_safe") is not True:
            raise ValueError("exploratory score analysis requires baseline-safe prompts")
        means: list[float] = []
        counts: list[int] = []
        for radius in radii:
            values = grouped.get((source, sample_id, radius), [])
            counts.append(len(values))
            means.append(sum(values) / len(values) if values else 0.0)
        curves.append(PromptScoreCurve(
            source=source,
            prompt_id=sample_id,
            fol=fol[(source, sample_id)],
            radii=radii,
            mean_unsafe_scores=tuple(means),
            accepted_direction_counts=tuple(counts),
        ))
    return tuple(curves)


def _endpoint_bands(curves: tuple[PromptScoreCurve, ...], selection: dict[str, Any]) -> list[dict[str, object]]:
    rows = []
    for source, values in sorted(selection["sources"].items()):
        for band in ("low", "middle", "high"):
            ids = set(values[band])
            aucs = [score_auc(curve) for curve in curves if curve.source == source and curve.prompt_id in ids]
            rows.append({"source": source, "band": band, "n": len(aucs), "mean_unsafe_score_auc": sum(aucs) / len(aucs)})
    return rows


def _display_source(source: str) -> str:
    return {"jailbound": "JailBound", "s_eval": "S-Eval"}.get(source, source)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# H1-v2 Continuous Unsafe-Score AUC",
        "",
        "> **Status: post-hoc exploratory.** This analysis does not replace the preregistered H1-v2 binary UTR-AUC endpoint.",
        "",
        "| Source | Spearman rho (FOL vs score-AUC) |",
        "|---|---:|",
    ]
    for source, rho in sorted(result["source_rho"].items()):
        lines.append(f"| {_display_source(source)} | {float(rho):.4f} |")
    lines.extend([
        "",
        f"Mean source-stratified rho: `{float(result['mean_rho']):.4f}`. ",
        f"Two-sided permutation p: `{float(result['permutation_pvalue']):.4f}`. ",
        "These values are hypothesis-generating only.",
        "",
        "| Source | Low FOL score-AUC | Middle FOL score-AUC | High FOL score-AUC |",
        "|---|---:|---:|---:|",
    ])
    grouped: dict[str, dict[str, float]] = {}
    for row in result["endpoint_bands_descriptive_only"]:
        grouped.setdefault(str(row["source"]), {})[str(row["band"])] = float(row["mean_unsafe_score_auc"])
    for source in sorted(grouped):
        values = grouped[source]
        lines.append(
            f"| {_display_source(source)} | {values['low']:.2%} | {values['middle']:.2%} | {values['high']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    root = ROOT / config.h1_v2.output_root
    selection = _read_object(root / "manifests" / "h1_v2_validation_selection.json", label="selection")
    calibration = _read_object(root / "locality_calibration.json", label="calibration")
    run = _read_object(root / "run_manifest.json", label="run manifest")
    config_hash = run.get("config_hash")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("exploratory score run manifest has no configuration hash")
    selected_ids = _selection_ids(selection, tuple(config.h1_v2.sources))
    radii = _radii(calibration)
    analysis = root / "analysis"
    contract_path = analysis / "exploratory_unsafe_score_auc_contract.json"
    expected_contract = _contract(config_hash=config_hash, selected_ids=selected_ids, radii=radii)
    if contract_path.exists():
        if _read_object(contract_path, label="contract") != expected_contract:
            raise ValueError("exploratory score contract does not match frozen inputs")
    else:
        atomic_write_json(contract_path, expected_contract)
    curves = _curves(root=root, config=config, selection=selection, selected_ids=selected_ids, radii=radii)
    association = source_stratified_score_association(
        curves,
        permutations=config.h1_v2.permutation_replicates,
        bootstrap_replicates=config.h1_v2.bootstrap_replicates,
        seed=config.base.run.seed,
    )
    result: dict[str, Any] = {
        "analysis_status": STATUS,
        "inference_scope": "hypothesis_generating_only",
        "endpoint": ENDPOINT,
        "source_rho": association.source_rho,
        "mean_rho": association.mean_rho,
        "permutation_pvalue": association.permutation_pvalue,
        "bootstrap_95_ci": [association.bootstrap_lower, association.bootstrap_upper],
        "endpoint_bands_descriptive_only": _endpoint_bands(curves, selection),
        "disclaimer": "This post-hoc exploratory analysis does not replace the preregistered H1-v2 binary UTR-AUC endpoint.",
    }
    atomic_write_json(analysis / "exploratory_unsafe_score_auc_results.json", result)
    (analysis / "exploratory_unsafe_score_auc.md").write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps({"status": STATUS, "artifact": str(analysis / "exploratory_unsafe_score_auc_results.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
