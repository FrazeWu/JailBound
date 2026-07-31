"""Analyse isolated H1-v3 endpoints and the numeric six-radius H2 extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchmark.safety_eval.config import load_h1_v3_config
from benchmark.safety_eval.fol_boundary import estimate_behavior_distance, fit_right_censored_cox
from benchmark.safety_eval.h1_v2_analysis import (
    PromptScoreCurve,
    PromptUtrCurve,
    score_auc,
    source_stratified_association,
    source_stratified_score_association,
    utr_auc,
)
from benchmark.safety_eval.h1_v3_runtime import (
    H1V3Paths,
    build_h1_v3_contract,
    frozen_h1_v2_selected_ids,
    frozen_h1_v2_selection,
    validate_h1_v3_contract,
)
from benchmark.safety_eval.io import atomic_write_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
METHOD_V2 = "fol_h1_v2"
METHOD_V3 = "fol_h1_v3"
STATUS = "follow_up_extension"
NEW_RADII = (0.4, 0.6)
ALL_RADII = (0.025, 0.05, 0.1, 0.2, 0.4, 0.6)


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"H1-v3 {label} is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"H1-v3 {label} must be an object")
    return payload


def _validate_source_contract(paths: H1V3Paths, *, radii: tuple[float, ...]) -> None:
    source_run = _read_object(paths.source_run_manifest(), label="source run manifest")
    source_hash = source_run.get("config_hash")
    if not isinstance(source_hash, str):
        raise ValueError("H1-v3 source run manifest has no configuration hash")
    selected_ids = frozen_h1_v2_selected_ids(paths.source_root)
    expected = build_h1_v3_contract(source_hash=source_hash, selected_ids=selected_ids, radii=radii)
    persisted = _read_object(paths.output_root / "source_contract.json", label="source contract")
    validate_h1_v3_contract(
        {key: persisted.get(key) for key in expected},
        source_hash=source_hash,
        selected_ids=selected_ids,
        radii=radii,
    )
    run = _read_object(paths.output_root / "run_manifest.json", label="run manifest")
    if run.get("analysis_status") != STATUS or run.get("source_h1_v2_config_hash") != source_hash:
        raise ValueError("H1-v3 run manifest differs from source contract")


def _directions(paths: H1V3Paths, *, root: Path, name: str, source: str) -> list[dict[str, object]]:
    rows = [row for row in read_jsonl(root / name) if row.get("source") == source]
    if not rows:
        raise ValueError(f"H1-v3 lacks {name} rows for {source}")
    return rows


def _baseline_labels(source_root: Path) -> dict[tuple[str, str], bool]:
    labels: dict[tuple[str, str], bool] = {}
    for row in read_jsonl(source_root / "baseline_labels.jsonl"):
        source, sample_id, baseline_safe = row.get("source"), row.get("sample_id"), row.get("baseline_safe")
        if isinstance(source, str) and isinstance(sample_id, str) and type(baseline_safe) is bool:
            labels[(source, sample_id)] = bool(baseline_safe)
    return labels


def _labels(root: Path, name: str) -> dict[str, bool]:
    labels = {}
    for row in read_jsonl(root / name):
        identifier, value = row.get("perturbation_id"), row.get("unsafe_label")
        if isinstance(identifier, str) and type(value) is bool:
            labels[identifier] = bool(value)
    return labels


def _scores(paths: H1V3Paths, *, config: Any, source: str) -> dict[str, float]:
    target_key = config.base.models.targets[0].key
    threshold = config.base.judging.primary.threshold
    scores: dict[str, float] = {}
    for row in read_jsonl(paths.judgment_ledger(target_key, source)):
        identifier, score = row.get("sample_id"), row.get("unsafe_score")
        if (
            isinstance(identifier, str)
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
            and row.get("checkpoint") == 0
            and row.get("threshold") == threshold
            and row.get("status") == "complete"
            and 0.0 <= float(score) <= 1.0
        ):
            scores[identifier] = float(score)
    return scores


def _curves(paths: H1V3Paths, config: Any) -> tuple[tuple[PromptScoreCurve, ...], tuple[PromptUtrCurve, ...]]:
    selection = frozen_h1_v2_selection(paths.source_root)
    selected_ids = frozen_h1_v2_selected_ids(paths.source_root)
    baselines = _baseline_labels(paths.source_root)
    labels = _labels(paths.output_root, "direction_labels.jsonl")
    score_curves: list[PromptScoreCurve] = []
    utr_curves: list[PromptUtrCurve] = []
    for source in config.h1_v3.sources:
        directions = _directions(paths, root=paths.output_root, name="selected_directions.jsonl", source=source)
        scores = _scores(paths, config=config, source=source)
        grouped_scores: dict[tuple[str, float], list[float]] = {}
        grouped_labels: dict[tuple[str, float], list[bool]] = {}
        for row in directions:
            sample_id, perturbation_id, radius = row.get("sample_id"), row.get("perturbation_id"), row.get("radius")
            if not isinstance(sample_id, str) or not isinstance(perturbation_id, str) or isinstance(radius, bool) or not isinstance(radius, (int, float)):
                raise ValueError("H1-v3 selected direction identity is invalid")
            if perturbation_id not in scores or perturbation_id not in labels:
                raise ValueError("H1-v3 direction judgment is incomplete")
            key = (sample_id, float(radius))
            grouped_scores.setdefault(key, []).append(scores[perturbation_id])
            grouped_labels.setdefault(key, []).append(labels[perturbation_id])
        for selected_source, sample_id in selected_ids:
            if selected_source != source:
                continue
            baseline_safe = baselines.get((source, sample_id))
            if baseline_safe is not True:
                raise ValueError("H1-v3 binary endpoint requires frozen baseline-safe prompts")
            score_values = [grouped_scores.get((sample_id, radius), []) for radius in NEW_RADII]
            label_values = [grouped_labels.get((sample_id, radius), []) for radius in NEW_RADII]
            if any(len(values) != config.h1_v3.accepted_directions for values in (*score_values, *label_values)):
                raise ValueError("H1-v3 endpoints require exactly 32 complete directions per prompt-radius")
            fol = selection[source]["fol_by_id"][sample_id]
            score_curves.append(PromptScoreCurve(
                source=source,
                prompt_id=sample_id,
                fol=fol,
                radii=NEW_RADII,
                mean_unsafe_scores=tuple(sum(values) / len(values) for values in score_values),
                accepted_direction_counts=(32, 32),
            ))
            utr_curves.append(PromptUtrCurve(
                source=source,
                prompt_id=sample_id,
                fol=fol,
                baseline_safe=True,
                radii=NEW_RADII,
                unsafe_transition_rates=tuple(sum(values) / len(values) for values in label_values),
                accepted_direction_counts=(32, 32),
            ))
    return tuple(score_curves), tuple(utr_curves)


def _bands(curves: tuple[PromptScoreCurve, ...], selection: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for source in ("jailbound", "s_eval"):
        for band in ("low", "middle", "high"):
            values = [score_auc(curve) for curve in curves if curve.source == source and curve.prompt_id in set(selection[source][band])]
            rows.append({"source": source, "band": band, "n": len(values), "mean_unsafe_score": sum(values) / len(values)})
    return rows


def _h2_rows(paths: H1V3Paths, config: Any) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    selection = frozen_h1_v2_selection(paths.source_root)
    selected_ids = frozen_h1_v2_selected_ids(paths.source_root)
    baselines = _baseline_labels(paths.source_root)
    labels_v2 = _labels(paths.source_root, "direction_labels.jsonl")
    labels_v3 = _labels(paths.output_root, "direction_labels.jsonl")
    rows: list[dict[str, object]] = []
    source_summary: dict[str, dict[str, object]] = {}
    for source in config.h1_v3.sources:
        directions = (
            _directions(paths, root=paths.source_root, name="selected_directions.jsonl", source=source)
            + _directions(paths, root=paths.output_root, name="selected_directions.jsonl", source=source)
        )
        grouped: dict[tuple[str, float], list[bool]] = {}
        for direction in directions:
            sample_id, perturbation_id, radius = direction.get("sample_id"), direction.get("perturbation_id"), direction.get("radius")
            if not isinstance(sample_id, str) or not isinstance(perturbation_id, str) or isinstance(radius, bool) or not isinstance(radius, (int, float)):
                raise ValueError("H1-v3 H2 direction identity is invalid")
            label = labels_v3.get(perturbation_id, labels_v2.get(perturbation_id))
            if label is None:
                raise ValueError("H1-v3 H2 direction label is incomplete")
            grouped.setdefault((sample_id, float(radius)), []).append(label)
        source_rows: list[dict[str, object]] = []
        for selected_source, sample_id in selected_ids:
            if selected_source != source:
                continue
            baseline_safe = baselines.get((source, sample_id))
            if baseline_safe is None:
                raise ValueError("H1-v3 H2 baseline label is incomplete")
            rates: dict[float, float] = {}
            for radius in ALL_RADII:
                values = grouped.get((sample_id, radius), [])
                if len(values) != config.h1_v3.accepted_directions:
                    raise ValueError("H1-v3 H2 requires exactly 32 labels at every radius")
                rates[radius] = sum(value != (not baseline_safe) for value in values) / len(values)
            distance = estimate_behavior_distance(rates)
            item = {
                "source": source,
                "prompt_id": sample_id,
                "fol": float(selection[source]["fol_by_id"][sample_id]),
                "d50": distance.estimate,
                "lower_bound": distance.lower,
                "upper_bound": distance.upper,
                "right_censored": distance.right_censored,
            }
            rows.append(item)
            source_rows.append(item)
        events = [not bool(row["right_censored"]) for row in source_rows]
        summary: dict[str, object] = {
            "usable_prompts": len(source_rows),
            "crossing_events": sum(events),
            "right_censored": len(source_rows) - sum(events),
        }
        try:
            estimate = fit_right_censored_cox(
                covariates=[float(row["fol"]) for row in source_rows],
                times=[float(row["d50"] if row["d50"] is not None else row["lower_bound"]) for row in source_rows],
                events=events,
            )
            summary.update({
                "hazard_ratio": estimate.hazard_ratio,
                "log_hazard_ratio": estimate.log_hazard_ratio,
                "pvalue": estimate.pvalue,
            })
        except ValueError as error:
            summary["cox_status"] = "inconclusive"
            summary["cox_reason"] = str(error)
        source_summary[source] = summary
    return rows, source_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_h1_v3_config(args.config)
    paths = H1V3Paths(ROOT / config.h1_v3.source_root, ROOT / config.h1_v3.output_root)
    _validate_source_contract(paths, radii=NEW_RADII)
    score_curves, utr_curves = _curves(paths, config)
    primary = source_stratified_score_association(
        score_curves,
        permutations=config.h1_v3.permutation_replicates,
        bootstrap_replicates=config.h1_v3.bootstrap_replicates,
        seed=config.base.run.seed,
    )
    secondary = source_stratified_association(
        utr_curves,
        permutations=config.h1_v3.permutation_replicates,
        bootstrap_replicates=config.h1_v3.bootstrap_replicates,
        seed=config.base.run.seed,
    )
    selection = frozen_h1_v2_selection(paths.source_root)
    analysis_root = paths.output_root / "analysis"
    atomic_write_json(analysis_root / "h1_v3_results.json", {
        "analysis_status": STATUS,
        "primary_endpoint": "mean_unsafe_score_over_r_0.4_and_0.6",
        "primary_test": "source_stratified_spearman_permutation_two_sided",
        "primary_source_rho": primary.source_rho,
        "primary_mean_rho": primary.mean_rho,
        "primary_permutation_pvalue": primary.permutation_pvalue,
        "primary_bootstrap_95_ci": [primary.bootstrap_lower, primary.bootstrap_upper],
        "secondary_endpoint": "unsafe_transition_rate_over_r_0.4_and_0.6",
        "secondary_source_rho": secondary.source_rho,
        "secondary_mean_rho": secondary.mean_rho,
        "secondary_permutation_pvalue": secondary.permutation_pvalue,
        "secondary_bootstrap_95_ci": [secondary.bootstrap_lower, secondary.bootstrap_upper],
        "descriptive_bands": _bands(score_curves, selection),
        "disclaimer": "This follow-up extension does not alter the preregistered H1-v2 binary UTR-AUC result.",
    })
    h2_rows, h2_summary = _h2_rows(paths, config)
    destination = analysis_root / "h1_v3_h2_d50.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "prompt_id", "fol", "d50", "lower_bound", "upper_bound", "right_censored"))
        writer.writeheader()
        writer.writerows(h2_rows)
    atomic_write_json(analysis_root / "h1_v3_h2_summary.json", {
        "analysis_status": STATUS,
        "radii": list(ALL_RADII),
        "sources": h2_summary,
        "disclaimer": "H2 uses numeric labels and frozen metadata only; remaining non-crossing prompts are right-censored.",
    })
    print(json.dumps({"h1_status": STATUS, "h2_prompt_count": len(h2_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
