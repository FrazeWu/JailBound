"""Write content-free FOL local-flip summaries from persisted metadata and labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.analysis import (
    FolClaimEvidence,
    FolFlipPrediction,
    decide_fol_claim,
    grouped_flip_prediction_comparison,
)
from benchmark.safety_eval.fol_boundary import (
    FolPerturbationOutcome,
    InterpolationPoint,
    MarginCalibrationRow,
    behavior_flip_rate,
    estimate_behavior_distance,
    exact_permutation_mean_difference,
    fit_right_censored_cox,
    fit_margin_calibration,
    summarize_interpolation_peaks,
    summarize_fol_bfr,
)
from benchmark.safety_eval.fol_records import resolved_terminal_payloads
from benchmark.safety_eval.io import atomic_write_json, read_jsonl
from benchmark.safety_eval.schema import JudgmentRecord, OptimizationRecord, RecordStatus


ROOT = Path(__file__).resolve().parents[1]


def _labels(
    root: Path,
    *,
    judge_key: str,
    target_key: str,
    threshold: float,
    sources: tuple[str, ...],
    method: str = "fol_boundary",
) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for source in sources:
        records = [
            JudgmentRecord.model_validate(row)
            for row in read_jsonl(root / "judgments" / judge_key / target_key / source / method / "records.jsonl")
        ]
        for record in records:
            if (method == "fol_boundary" and record.checkpoint != 0) or record.threshold != threshold:
                continue
            if record.sample_id in labels:
                raise ValueError("duplicate FOL judgment label")
            if record.status is RecordStatus.complete:
                labels[record.sample_id] = record.unsafe_label
    return labels


def _write_h3(
    destination: Path,
    *,
    root: Path,
    primary_key: str,
    primary_threshold: float,
    target_key: str,
    sources: tuple[str, ...],
    minimum_valid: int,
    minimum_paths: int,
    permutations: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    metadata = read_jsonl(root / "interpolation_points.jsonl")
    labels = _labels(
        root,
        judge_key=primary_key,
        target_key=target_key,
        threshold=primary_threshold,
        sources=sources,
        method="fol_interpolation",
    )
    by_source_path: dict[tuple[str, str], list[InterpolationPoint]] = {}
    for row in metadata:
        source, path_id, interpolation_id = row.get("source"), row.get("path_id"), row.get("interpolation_id")
        point_index = row.get("point_index")
        semantic_accepted = row.get("semantic_accepted")
        fol, curvature = row.get("fol"), row.get("curvature")
        if source not in sources:
            continue
        if (
            not isinstance(source, str)
            or not isinstance(path_id, str)
            or not isinstance(interpolation_id, str)
            or isinstance(point_index, bool)
            or not isinstance(point_index, int)
            or type(semantic_accepted) is not bool
            or isinstance(fol, bool)
            or not isinstance(fol, (int, float))
            or isinstance(curvature, bool)
            or not isinstance(curvature, (int, float))
        ):
            raise ValueError("FOL interpolation metadata is invalid")
        if semantic_accepted and interpolation_id not in labels:
            raise ValueError("semantic-valid FOL interpolation point lacks a primary label")
        by_source_path.setdefault((source, path_id), []).append(InterpolationPoint(
            path_id=path_id,
            point_index=point_index,
            semantic_accepted=semantic_accepted,
            unsafe_label=labels.get(interpolation_id, False),
            fol=float(fol),
            curvature=float(curvature),
        ))
    results: dict[str, dict[str, object]] = {}
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "Source", "Valid paths", "Crossing paths", "FOL mean distance", "Curvature mean distance",
            "Random mean distance", "FOL vs random p", "FOL vs curvature p", "Status", "Provenance",
        ))
        writer.writeheader()
        for source in sources:
            paths = [rows for (row_source, _path_id), rows in sorted(by_source_path.items()) if row_source == source]
            summary = summarize_interpolation_peaks(
                paths,
                minimum_valid=minimum_valid,
                minimum_paths=minimum_paths,
                permutations=permutations,
                seed=seed,
            )
            status = "ready" if summary.fol_mean_distance is not None else "inconclusive"
            results[source] = {
                "valid_paths": summary.valid_path_count,
                "crossing_paths": summary.crossing_path_count,
                "status": status,
            }
            writer.writerow({
                "Source": source,
                "Valid paths": summary.valid_path_count,
                "Crossing paths": summary.crossing_path_count,
                "FOL mean distance": summary.fol_mean_distance,
                "Curvature mean distance": summary.curvature_mean_distance,
                "Random mean distance": summary.random_mean_distance,
                "FOL vs random p": summary.fol_vs_random_pvalue,
                "FOL vs curvature p": summary.fol_vs_curvature_pvalue,
                "Status": status,
                "Provenance": "new_run",
            })
    return results


def _main_margin_threshold(
    root: Path,
    *,
    primary_key: str,
    primary_threshold: float,
    target_key: str,
    sources: tuple[str, ...],
) -> float:
    rows: list[MarginCalibrationRow] = []
    main_root = root.parent
    for source in sources:
        margins = {
            record.sample_id: record.internal_margin
            for record in (
                OptimizationRecord.model_validate(payload)
                for payload in read_jsonl(main_root / "optimization" / source / "jailbound_o_plus" / "records.jsonl")
            )
            if record.checkpoint == 100
            and record.status is RecordStatus.complete
            and record.internal_margin is not None
        }
        labels = {
            record.sample_id: record.unsafe_label
            for record in (
                JudgmentRecord.model_validate(payload)
                for payload in read_jsonl(main_root / "judgments" / primary_key / target_key / source / "jailbound_o_plus" / "records.jsonl")
            )
            if record.checkpoint == 100
            and record.threshold == primary_threshold
            and record.status is RecordStatus.complete
        }
        if set(margins) != set(labels):
            raise ValueError(f"main O+ margin calibration records are incomplete for {source}")
        rows.extend(MarginCalibrationRow(f"{source}:{sample_id}", float(margin), labels[sample_id]) for sample_id, margin in margins.items())
    calibration = fit_margin_calibration(rows, excluded_ids=set())
    if calibration.threshold is None:
        raise ValueError("main O+ margin calibration never reaches the behavior threshold")
    return calibration.threshold


def _write_h4(
    destination: Path,
    *,
    root: Path,
    controls_root: Path,
    primary_key: str,
    primary_threshold: float,
    target_key: str,
    sources: tuple[str, ...],
    folds: int,
    seed: int,
) -> dict[str, object]:
    fieldnames = (
        "Scope", "Held-out rows", "Controls AUROC", "Controls-plus-FOL AUROC", "Delta AUROC",
        "Controls AUPRC", "Controls-plus-FOL AUPRC", "Delta AUPRC", "Controls Brier",
        "Controls-plus-FOL Brier", "Delta Brier", "Controls ECE", "Controls-plus-FOL ECE",
        "Delta ECE", "Margin threshold", "Status", "Provenance",
    )
    controls = {
        (str(row.get("source")), str(row.get("sample_id"))): row
        for row in read_jsonl(controls_root / "controls.jsonl")
        if isinstance(row.get("source"), str) and isinstance(row.get("sample_id"), str)
    }
    if not controls:
        status = {"status": "inconclusive", "reason": "no_controls"}
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"Scope": "combined", "Status": status["status"], "Provenance": "new_run"})
        return status
    try:
        margin_threshold = _main_margin_threshold(
            root, primary_key=primary_key, primary_threshold=primary_threshold,
            target_key=target_key, sources=sources,
        )
        labels = _labels(
            root, judge_key=primary_key, target_key=target_key,
            threshold=primary_threshold, sources=sources,
        )
        metadata = [row for row in read_jsonl(root / "selected_perturbations.jsonl") if row.get("source") in sources]
        baseline: dict[tuple[str, str], str] = {}
        predictions: list[FolFlipPrediction] = []
        for row in metadata:
            source, sample_id, candidate_id, kind = row.get("source"), row.get("sample_id"), row.get("perturbation_id"), row.get("kind")
            if not all(isinstance(value, str) and value for value in (source, sample_id, candidate_id, kind)):
                raise ValueError("FOL control metadata is invalid")
            if kind == "baseline":
                baseline[(source, sample_id)] = candidate_id
        for row in metadata:
            source, sample_id, candidate_id, kind, radius = (
                row.get("source"), row.get("sample_id"), row.get("perturbation_id"), row.get("kind"), row.get("radius"),
            )
            if kind != "perturbation":
                continue
            if (
                not isinstance(source, str)
                or not isinstance(sample_id, str)
                or not isinstance(candidate_id, str)
                or isinstance(radius, bool)
                or not isinstance(radius, (int, float))
            ):
                raise ValueError("FOL control perturbation metadata is invalid")
            baseline_id = baseline.get((source, sample_id))
            control = controls.get((source, sample_id))
            if baseline_id is None or control is None or candidate_id not in labels or baseline_id not in labels:
                continue
            values = tuple(control.get(field) for field in (
                "attack_loss", "internal_margin", "prompt_length", "perplexity", "curvature", "roughness", "semantic_acceptance_rate",
            ))
            fol = control.get("fol")
            if (
                isinstance(fol, bool)
                or not isinstance(fol, (int, float))
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)
            ):
                raise ValueError("FOL control values are invalid")
            attack_loss, internal_margin, prompt_length, perplexity, curvature, roughness, acceptance_rate = (float(value) for value in values)
            predictions.append(FolFlipPrediction(
                prompt_id=f"{source}:{sample_id}",
                flipped=labels[candidate_id] is not labels[baseline_id],
                fol=float(fol),
                controls=(
                    attack_loss,
                    abs(internal_margin - margin_threshold),
                    prompt_length,
                    perplexity,
                    curvature,
                    roughness,
                    acceptance_rate,
                    float(radius),
                ),
            ))
        comparison = grouped_flip_prediction_comparison(predictions, folds=folds, seed=seed)
        row = {
            "Scope": "combined",
            "Held-out rows": comparison.held_out_rows,
            "Controls AUROC": comparison.controls_auroc,
            "Controls-plus-FOL AUROC": comparison.fol_auroc,
            "Delta AUROC": comparison.delta_auroc,
            "Controls AUPRC": comparison.controls_auprc,
            "Controls-plus-FOL AUPRC": comparison.fol_auprc,
            "Delta AUPRC": comparison.delta_auprc,
            "Controls Brier": comparison.controls_brier,
            "Controls-plus-FOL Brier": comparison.fol_brier,
            "Delta Brier": comparison.delta_brier,
            "Controls ECE": comparison.controls_ece,
            "Controls-plus-FOL ECE": comparison.fol_ece,
            "Delta ECE": comparison.delta_ece,
            "Margin threshold": margin_threshold,
            "Status": "ready",
            "Provenance": "new_run",
        }
        status = {"status": "ready", "held_out_rows": comparison.held_out_rows}
    except ValueError as error:
        row = {"Scope": "combined", "Status": "inconclusive", "Provenance": "new_run"}
        status = {"status": "inconclusive", "reason": type(error).__name__}
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return status


def _outcomes(
    root: Path,
    *,
    primary_key: str,
    primary_threshold: float,
    secondary_key: str | None,
    secondary_threshold: float | None,
    target_key: str,
    sources: tuple[str, ...],
) -> tuple[tuple[FolPerturbationOutcome, ...], dict[str, int]]:
    metadata = [row for row in read_jsonl(root / "selected_perturbations.jsonl") if row.get("source") in sources]
    primary = _labels(
        root, judge_key=primary_key, target_key=target_key,
        threshold=primary_threshold, sources=sources,
    )
    secondary = (
        _labels(root, judge_key=secondary_key, target_key=target_key, threshold=secondary_threshold, sources=sources)
        if secondary_key is not None and secondary_threshold is not None
        else {}
    )
    baseline: dict[tuple[str, str], dict[str, str]] = {}
    perturbations: list[dict[str, object]] = []
    for row in metadata:
        source, sample_id, perturbation_id, kind = (
            row.get("source"), row.get("sample_id"), row.get("perturbation_id"), row.get("kind"),
        )
        if not all(isinstance(value, str) and value for value in (source, sample_id, perturbation_id, kind)):
            raise ValueError("FOL selected-perturbation metadata is invalid")
        key = (source, sample_id)
        if kind == "baseline":
            if key in baseline:
                raise ValueError("duplicate FOL baseline metadata")
            band = row.get("band")
            if not isinstance(band, str) or not band:
                raise ValueError("FOL baseline metadata has no band")
            baseline[key] = {"perturbation_id": perturbation_id, "band": band}
        elif kind == "perturbation":
            perturbations.append(row)
        else:
            raise ValueError("unknown FOL metadata kind")
    output: list[FolPerturbationOutcome] = []
    failures = {"primary": 0, "secondary": 0, "both": 0}
    selected_groups: set[tuple[str, str, float]] = set()
    for row in perturbations:
        source, sample_id = row["source"], row["sample_id"]
        if not isinstance(source, str) or not isinstance(sample_id, str):
            raise ValueError("FOL perturbation identity is invalid")
        baseline_id = baseline.get((source, sample_id), {}).get("perturbation_id")
        perturbation_id = row.get("perturbation_id")
        if not isinstance(baseline_id, str) or not isinstance(perturbation_id, str):
            raise ValueError("FOL perturbation baseline is missing")
        primary_ready = baseline_id in primary and perturbation_id in primary
        secondary_ready = secondary_key is None or (baseline_id in secondary and perturbation_id in secondary)
        if not primary_ready:
            failures["primary"] += 1
        if secondary_key is not None and not secondary_ready:
            failures["secondary"] += 1
        if not primary_ready or not secondary_ready:
            failures["both"] += 1
            continue
        band, radius = row.get("band"), row.get("radius")
        if not isinstance(band, str) or isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise ValueError("FOL perturbation metadata lacks band or radius")
        output.append(FolPerturbationOutcome(
            source=source,
            sample_id=sample_id,
            band=band,
            radius=float(radius),
            accepted=True,
            primary_label=primary[perturbation_id],
            primary_baseline_label=primary[baseline_id],
            secondary_label=secondary[perturbation_id] if secondary_key is not None else None,
            secondary_baseline_label=secondary[baseline_id] if secondary_key is not None else None,
        ))
        selected_groups.add((source, sample_id, float(radius)))

    # Keep source/band/radius denominators visible even when semantic filtering
    # leaves a prompt with zero executable directions.
    for schedule in read_jsonl(root / "perturbation_schedule.jsonl"):
        source, sample_id, radius = schedule.get("source"), schedule.get("sample_id"), schedule.get("radius")
        if (
            not isinstance(source, str)
            or not isinstance(sample_id, str)
            or isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or source not in sources
        ):
            continue
        group = (source, sample_id, float(radius))
        if group in selected_groups:
            continue
        baseline_row = baseline.get((source, sample_id))
        if baseline_row is None:
            continue
        baseline_id = baseline_row["perturbation_id"]
        if baseline_id not in primary or (secondary_key is not None and baseline_id not in secondary):
            continue
        output.append(FolPerturbationOutcome(
            source=source,
            sample_id=sample_id,
            band=baseline_row["band"],
            radius=float(radius),
            accepted=False,
            primary_label=primary[baseline_id],
            primary_baseline_label=primary[baseline_id],
            secondary_label=secondary[baseline_id] if secondary_key is not None else None,
            secondary_baseline_label=secondary[baseline_id] if secondary_key is not None else None,
        ))
        selected_groups.add(group)
    return tuple(output), failures


def _write_bfr(destination: Path, rows: tuple[FolPerturbationOutcome, ...], minimum_accepted: int) -> None:
    summaries = summarize_fol_bfr(rows, minimum_accepted=minimum_accepted)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "Source", "Band", "Radius", "Judge", "Prompts", "Eligible prompts",
            "Sparse prompts", "Accepted directions", "Mean BFR", "Provenance",
        ))
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "Source": row.source,
                "Band": row.band,
                "Radius": row.radius,
                "Judge": row.judge_key,
                "Prompts": row.prompt_count,
                "Eligible prompts": row.eligible_prompt_count,
                "Sparse prompts": row.sparse_prompt_count,
                "Accepted directions": row.accepted_direction_count,
                "Mean BFR": row.mean_bfr,
                "Provenance": "new_run",
            })


def _d50_rows(
    rows: tuple[FolPerturbationOutcome, ...], minimum_accepted: int
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], dict[float, list[bool]]] = {}
    for row in rows:
        labels = [("primary", row.primary_label, row.primary_baseline_label)]
        if row.secondary_label is not None and row.secondary_baseline_label is not None:
            labels.append(("secondary", row.secondary_label, row.secondary_baseline_label))
        for judge, label, baseline in labels:
            groups.setdefault((row.source, row.sample_id, row.band, judge), {}).setdefault(row.radius, []).append(label is not baseline)
    output: list[dict[str, object]] = []
    for (source, sample_id, band, judge), flips_by_radius in sorted(groups.items()):
        rates = {
            radius: value.rate
            for radius, flips in flips_by_radius.items()
            if (value := behavior_flip_rate(flips, minimum_accepted=minimum_accepted)).rate is not None
        }
        if len(rates) < 2:
            continue
        distance = estimate_behavior_distance(rates)
        output.append({
            "source": source,
            "sample_id": sample_id,
            "band": band,
            "judge": judge,
            "usable_radii": len(rates),
            "d50": distance.estimate,
            "lower_bound": distance.lower,
            "right_censored": distance.right_censored,
        })
    return output


def _write_d50(destination: Path, rows: list[dict[str, object]]) -> None:
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "Source", "Sample ID", "Band", "Judge", "Usable radii", "d50", "Lower bound",
            "Right censored", "Provenance",
        ))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Source": row["source"],
                "Sample ID": row["sample_id"],
                "Band": row["band"],
                "Judge": row["judge"],
                "Usable radii": row["usable_radii"],
                "d50": row["d50"],
                "Lower bound": row["lower_bound"],
                "Right censored": row["right_censored"],
                "Provenance": "new_run",
            })


def _terminal_fol(root: Path, sources: tuple[str, ...]) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for source in sources:
        for payload in resolved_terminal_payloads(root, source).values():
            record = OptimizationRecord.model_validate(payload)
            if record.checkpoint != 100 or record.status is not RecordStatus.complete:
                continue
            if record.fol is None:
                raise ValueError("FOL terminal record has no FOL value")
            key = (source, record.sample_id)
            if key in values:
                raise ValueError("duplicate FOL terminal record")
            values[key] = float(record.fol)
    return values


def _write_h2(destination: Path, d50_rows: list[dict[str, object]], fol_by_sample: dict[tuple[str, str], float]) -> None:
    import math

    from scipy.stats import spearmanr

    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in d50_rows:
        source, sample_id, judge = row["source"], row["sample_id"], row["judge"]
        if not all(isinstance(value, str) for value in (source, sample_id, judge)):
            raise ValueError("FOL d50 row has invalid identity")
        groups.setdefault((source, judge), []).append(row)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "Source", "Judge", "Usable samples", "Right censored", "Uncensored samples",
            "FOL-d50 Spearman rho", "Spearman p", "Cox events",
            "Cox HR per log-FOL SD", "Cox p", "Provenance",
        ))
        writer.writeheader()
        for (source, judge), rows in sorted(groups.items()):
            uncensored = [row for row in rows if row["d50"] is not None]
            fol = [fol_by_sample[(source, str(row["sample_id"]))] for row in uncensored]
            d50 = [float(row["d50"]) for row in uncensored]
            if len(uncensored) >= 3 and len(set(fol)) > 1 and len(set(d50)) > 1:
                statistic = spearmanr(fol, d50)
                rho, pvalue = float(statistic.statistic), float(statistic.pvalue)
            else:
                rho = pvalue = None
            log_fol = [math.log(fol_by_sample[(source, str(row["sample_id"]))]) for row in rows]
            mean_log_fol = sum(log_fol) / len(log_fol)
            variance = sum((value - mean_log_fol) ** 2 for value in log_fol) / len(log_fol)
            if variance > 0.0:
                scale = math.sqrt(variance)
                try:
                    cox = fit_right_censored_cox(
                        covariates=[(value - mean_log_fol) / scale for value in log_fol],
                        times=[float(row["d50"]) if row["d50"] is not None else float(row["lower_bound"]) for row in rows],
                        events=[row["d50"] is not None for row in rows],
                    )
                    cox_events, cox_hr, cox_pvalue = cox.event_count, cox.hazard_ratio, cox.pvalue
                except ValueError:
                    cox_events = cox_hr = cox_pvalue = None
            else:
                cox_events = cox_hr = cox_pvalue = None
            writer.writerow({
                "Source": source,
                "Judge": judge,
                "Usable samples": len(rows),
                "Right censored": sum(row["right_censored"] is True for row in rows),
                "Uncensored samples": len(uncensored),
                "FOL-d50 Spearman rho": rho,
                "Spearman p": pvalue,
                "Cox events": cox_events,
                "Cox HR per log-FOL SD": cox_hr,
                "Cox p": cox_pvalue,
                "Provenance": "new_run",
            })


def _write_h1(destination: Path, rows: tuple[FolPerturbationOutcome, ...], minimum_accepted: int) -> None:
    groups: dict[tuple[str, str, float, str], dict[str, list[float]]] = {}
    direction_groups: dict[tuple[str, str, str, float, str], list[bool]] = {}
    for row in rows:
        labels = [("primary", row.primary_label, row.primary_baseline_label)]
        if row.secondary_label is not None and row.secondary_baseline_label is not None:
            labels.append(("secondary", row.secondary_label, row.secondary_baseline_label))
        for judge, label, baseline in labels:
            direction_groups.setdefault(
                (row.source, row.sample_id, row.band, row.radius, judge), []
            ).append(label is not baseline)
    for (source, _sample_id, band, radius, judge), flips in direction_groups.items():
        bfr = behavior_flip_rate(flips, minimum_accepted=minimum_accepted)
        if bfr.rate is not None and band in {"low", "high"}:
            groups.setdefault((source, band, radius, judge), {}).setdefault(band, []).append(bfr.rate)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "Source", "Radius", "Judge", "Low usable prompts", "High usable prompts",
            "Low mean BFR", "High mean BFR", "High-minus-low", "Exact permutation p", "Provenance",
        ))
        writer.writeheader()
        cells = sorted({(source, radius, judge) for source, _band, radius, judge in groups})
        for source, radius, judge in cells:
            low = groups.get((source, "low", radius, judge), {}).get("low", [])
            high = groups.get((source, "high", radius, judge), {}).get("high", [])
            difference, pvalue = (exact_permutation_mean_difference(low, high) if low and high else (None, None))
            writer.writerow({
                "Source": source,
                "Radius": radius,
                "Judge": judge,
                "Low usable prompts": len(low),
                "High usable prompts": len(high),
                "Low mean BFR": (sum(low) / len(low)) if low else None,
                "High mean BFR": (sum(high) / len(high)) if high else None,
                "High-minus-low": difference,
                "Exact permutation p": pvalue,
                "Provenance": "new_run",
            })


def _claim_payload(
    *,
    h3: dict[str, dict[str, object]],
    failures: dict[str, int],
    usable_directions: int,
    minimum_paths: int,
) -> dict[str, object]:
    sources = tuple(sorted(h3))
    valid_paths = {
        source: int(h3[source].get("valid_paths", 0))
        for source in sources
    }
    if usable_directions < 0 or minimum_paths < 1 or any(value < 0 for value in failures.values()):
        raise ValueError("FOL claim inputs are invalid")
    observed_minimum = min(valid_paths.values(), default=0)
    evidence = FolClaimEvidence(
        h1=False,
        h2=False,
        h3=all(h3[source].get("status") == "ready" for source in sources),
        h4=False,
        secondary_same_direction=False,
        valid_paths=observed_minimum,
        usable_fraction=1.0 if usable_directions else 0.0,
        band_acceptance_difference=0.0,
        h1_interval_width=float("inf"),
        h2_interval_width=float("inf"),
    )
    decision = decide_fol_claim(evidence)
    reason = "interpolation_underpowered" if observed_minimum < minimum_paths else "hypothesis_evidence_incomplete"
    return {
        "decision": decision,
        "provenance": "new_run",
        "reason": reason,
        "quality_gates": {
            "minimum_valid_paths_per_source": minimum_paths,
            **{f"{source}_valid_paths": valid_paths[source] for source in sources},
            "interpolation_gate_passed": observed_minimum >= minimum_paths,
        },
        "execution": {
            "dual_judge_usable_directions": usable_directions,
            "judgment_failures": sum(failures.values()),
        },
        "interpretation": "No boundary-proximity claim is supported when the pre-registered interpolation quality gate is not met.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fol-root", type=Path)
    parser.add_argument("--state-fol-root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.fol_root or (ROOT / config.run.output_root / "fol_boundary")
    if not root.is_absolute():
        root = ROOT / root
    state_root = args.state_fol_root or root
    if not state_root.is_absolute():
        state_root = ROOT / state_root
    rows, failures = _outcomes(
        root,
        primary_key=config.judging.primary.key,
        primary_threshold=config.judging.primary.threshold,
        secondary_key=None if args.primary_only else config.judging.secondary.key,
        secondary_threshold=None if args.primary_only else config.judging.secondary.threshold,
        target_key=config.models.targets[0].key,
        sources=tuple(config.fol.sources),
    )
    analysis = args.analysis_dir or (root / "analysis")
    if not analysis.is_absolute():
        analysis = ROOT / analysis
    analysis.mkdir(parents=True, exist_ok=True)
    _write_bfr(analysis / "fol_bfr.csv", rows, config.fol.minimum_accepted_directions)
    d50_rows = _d50_rows(rows, config.fol.minimum_accepted_directions)
    _write_d50(analysis / "fol_d50.csv", d50_rows)
    _write_h2(
        analysis / "fol_h2_d50.csv",
        d50_rows,
        _terminal_fol(state_root, tuple(config.fol.sources)),
    )
    _write_h1(analysis / "fol_h1_bfr.csv", rows, config.fol.minimum_accepted_directions)
    h3 = _write_h3(
        analysis / "fol_h3_interpolation.csv",
        root=root,
        primary_key=config.judging.primary.key,
        primary_threshold=config.judging.primary.threshold,
        target_key=config.models.targets[0].key,
        sources=tuple(config.fol.sources),
        minimum_valid=config.fol.minimum_valid_interpolation_points,
        minimum_paths=config.fol.minimum_valid_paths,
        permutations=config.fol.permutation_replicates,
        seed=config.run.seed,
    )
    h4 = _write_h4(
        analysis / "fol_h4_controls.csv",
        root=root,
        controls_root=state_root,
        primary_key=config.judging.primary.key,
        primary_threshold=config.judging.primary.threshold,
        target_key=config.models.targets[0].key,
        sources=tuple(config.fol.sources),
        folds=5,
        seed=config.run.seed,
    )
    atomic_write_json(analysis / "fol_execution_quality.json", {
        "dual_judge_usable_directions": len(rows),
        "judgment_failures": failures,
        "interpolation": h3,
        "controls": h4,
    })
    atomic_write_json(
        analysis / "fol_boundary_claim.json",
        _claim_payload(
            h3=h3,
            failures=failures,
            usable_directions=len(rows),
            minimum_paths=config.fol.minimum_valid_paths,
        ),
    )
    print(json.dumps({"directions": len(rows), "judgment_failures": failures}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
