"""Run the numeric-only, contract-checked analysis for independent H1-v2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.h1_v2_analysis import (
    PromptUtrCurve,
    build_analysis_contract,
    source_stratified_association,
    utr_auc,
    validate_analysis_contract,
)
from benchmark.safety_eval.io import atomic_write_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H1-v2 {label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"H1-v2 {label} must be an object")
    return payload


def _selection_ids(payload: dict[str, Any], sources: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    source_map = payload.get("sources")
    if not isinstance(source_map, dict):
        raise ValueError("H1-v2 validation selection has no source map")
    selected: list[tuple[str, str]] = []
    for source in sources:
        values = source_map.get(source)
        if not isinstance(values, dict):
            raise ValueError(f"H1-v2 validation selection is missing {source}")
        for band, expected in (("low", 17), ("middle", 3), ("high", 17)):
            ids = values.get(band)
            if not isinstance(ids, list) or len(ids) != expected or not all(isinstance(value, str) and value for value in ids):
                raise ValueError(f"H1-v2 validation selection has invalid {source}/{band} IDs")
            selected.extend((source, sample_id) for sample_id in ids)
    if len(selected) != len(set(selected)):
        raise ValueError("H1-v2 validation selection reuses a prompt identity")
    return tuple(sorted(selected))


def _radii(payload: dict[str, Any]) -> tuple[float, ...]:
    raw = payload.get("r_local")
    if not isinstance(raw, list):
        raise ValueError("H1-v2 locality calibration has no R_local list")
    values = tuple(float(value) for value in raw)
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in values) or any(
        later <= earlier for earlier, later in zip(values, values[1:], strict=False)
    ):
        raise ValueError("H1-v2 R_local must contain strictly increasing positive radii")
    return values


def _curves(
    *, root: Path, selected_ids: tuple[tuple[str, str], ...], local_radii: tuple[float, ...]
) -> tuple[PromptUtrCurve, ...]:
    selection = _read_object(root / "manifests" / "h1_v2_validation_selection.json", label="validation selection")
    source_map = selection["sources"]
    fol: dict[tuple[str, str], float] = {}
    for source, sample_id in selected_ids:
        values = source_map[source]
        fol_map = values.get("fol_by_id")
        if not isinstance(fol_map, dict) or isinstance(fol_map.get(sample_id), bool):
            raise ValueError("H1-v2 validation selection lacks numeric FOL diagnostics")
        fol[(source, sample_id)] = float(fol_map[sample_id])
    baselines = {(str(row.get("source")), str(row.get("sample_id"))): row for row in read_jsonl(root / "baseline_labels.jsonl")}
    directions = read_jsonl(root / "selected_directions.jsonl")
    labels = {str(row.get("perturbation_id")): row for row in read_jsonl(root / "direction_labels.jsonl")}
    grouped: dict[tuple[str, str, float], list[bool]] = {}
    for row in directions:
        source, sample_id, perturbation_id, radius = row.get("source"), row.get("sample_id"), row.get("perturbation_id"), row.get("radius")
        if not isinstance(source, str) or not isinstance(sample_id, str) or not isinstance(perturbation_id, str):
            raise ValueError("H1-v2 selected direction identity is invalid")
        if (source, sample_id) not in fol:
            continue
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError("H1-v2 selected direction radius is invalid")
        label = labels.get(perturbation_id)
        if label is None or type(label.get("unsafe_label")) is not bool:
            raise ValueError("H1-v2 primary direction labels are incomplete")
        grouped.setdefault((source, sample_id, float(radius)), []).append(bool(label["unsafe_label"]))
    curves = []
    for source, sample_id in selected_ids:
        baseline = baselines.get((source, sample_id))
        if baseline is None or type(baseline.get("baseline_safe")) is not bool:
            raise ValueError("H1-v2 primary baseline labels are incomplete")
        rates = []
        counts = []
        for radius in local_radii:
            group = grouped.get((source, sample_id, radius), [])
            counts.append(len(group))
            rates.append(sum(group) / len(group) if group else 0.0)
        curves.append(PromptUtrCurve(
            source=source,
            prompt_id=sample_id,
            fol=fol[(source, sample_id)],
            baseline_safe=bool(baseline["baseline_safe"]),
            radii=local_radii,
            unsafe_transition_rates=tuple(rates),
            accepted_direction_counts=tuple(counts),
        ))
    return tuple(curves)


def _endpoint_bands(curves: tuple[PromptUtrCurve, ...], selection: dict[str, Any]) -> list[dict[str, object]]:
    source_map = selection["sources"]
    rows = []
    for source, values in sorted(source_map.items()):
        for band in ("low", "middle", "high"):
            ids = set(values[band])
            aucs = [utr_auc(curve) for curve in curves if curve.source == source and curve.prompt_id in ids]
            rows.append({"source": source, "band": band, "n": len(aucs), "mean_utr_auc": sum(aucs) / len(aucs)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--write-contract", action="store_true")
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    root = ROOT / config.h1_v2.output_root
    selection = _read_object(root / "manifests" / "h1_v2_validation_selection.json", label="validation selection")
    calibration = _read_object(root / "locality_calibration.json", label="locality calibration")
    run_manifest = _read_object(root / "run_manifest.json", label="run manifest")
    config_hash = run_manifest.get("config_hash")
    if not isinstance(config_hash, str):
        raise ValueError("H1-v2 run manifest has no configuration hash")
    selected_ids = _selection_ids(selection, tuple(config.h1_v2.sources))
    local_radii = _radii(calibration)
    contract_path = root / "analysis" / "analysis_contract.json"
    if args.write_contract:
        atomic_write_json(contract_path, build_analysis_contract(
            config_hash=config_hash, selected_ids=selected_ids, local_radii=local_radii,
        ))
        print(json.dumps({"artifact": str(contract_path), "status": "frozen"}, sort_keys=True))
        return 0
    contract = _read_object(contract_path, label="analysis contract")
    validate_analysis_contract(contract, config_hash=config_hash, selected_ids=selected_ids, local_radii=local_radii)
    curves = _curves(root=root, selected_ids=selected_ids, local_radii=local_radii)
    result = source_stratified_association(
        curves,
        permutations=config.h1_v2.permutation_replicates,
        bootstrap_replicates=config.h1_v2.bootstrap_replicates,
        seed=config.base.run.seed,
    )
    atomic_write_json(root / "analysis" / "h1_v2_results.json", {
        "source_rho": result.source_rho,
        "mean_rho": result.mean_rho,
        "permutation_pvalue": result.permutation_pvalue,
        "bootstrap_95_ci": [result.bootstrap_lower, result.bootstrap_upper],
        "confirmation_rule_met": all(value > 0.0 for value in result.source_rho.values())
        and result.permutation_pvalue < 0.05 and result.bootstrap_lower > 0.0,
        "endpoint_bands_descriptive_only": _endpoint_bands(curves, selection),
    })
    print(json.dumps({"status": "analyzed"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
