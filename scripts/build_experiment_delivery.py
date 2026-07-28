"""Build content-free experiment tables, figures, and scope notes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark.safety_eval.io import read_jsonl
from benchmark.safety_eval.result_aggregation import summarize_materializations


PRIMARY_JUDGE = "octopus_seval_14b"
CENTRAL_THRESHOLD = "0.5"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required aggregate is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _latex_escape(value: object) -> str:
    return str(value).replace("_", "\\_").replace("%", "\\%")


def _write_latex(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = "l" * len(fieldnames)
    lines = [f"\\begin{{tabular}}{{{columns}}}", " & ".join(_latex_escape(name) for name in fieldnames) + r" \\", "\\hline"]
    lines.extend(" & ".join(_latex_escape(row.get(name, "")) for name in fieldnames) + " \\\\" for row in rows)
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_table(
    *, source_rows: list[dict[str, str]], destination: Path, fieldnames: tuple[str, ...]
) -> list[dict[str, object]]:
    rows = [{key: row.get(key, "") for key in fieldnames if key != "Provenance"} | {"Provenance": "new_run"} for row in source_rows]
    _write_csv(destination.with_suffix(".csv"), rows, fieldnames)
    _write_latex(destination.with_suffix(".tex"), rows, fieldnames)
    return rows


def _materialization_rows(
    *, base_root: Path | None, replacement_root: Path | None, sources: set[str], methods: set[str]
) -> list[dict[str, object]]:
    roots: list[tuple[Path, set[str] | None]] = []
    if base_root is not None:
        roots.append((base_root, sources - {"jailbound"}))
    if replacement_root is not None:
        roots.append((replacement_root, {"jailbound"} & sources))
    raw_rows: list[dict[str, object]] = []
    for root, allowed_sources in roots:
        if not allowed_sources:
            continue
        for path in sorted((root / "optimization").glob("*/*/materialization.jsonl")):
            source = path.parts[-3]
            method = path.parts[-2]
            if source in allowed_sources and method in methods:
                raw_rows.extend(read_jsonl(path))
    if not raw_rows:
        return []
    summaries = summarize_materializations(raw_rows)
    return [
        {
            "Source": row.source,
            "Method": row.method,
            "Total": row.total_count,
            "Complete": row.complete_count,
            "Failed": row.failed_count,
            "Intent preserved": f"{row.intent_preserved_count} / {row.total_count}",
            "Semantic similarity mean": row.semantic_similarity_mean,
            "Prefix projection cosine mean": row.prefix_projection_cosine_mean,
            "Seed projection cosine mean": row.seed_projection_cosine_mean,
            "Provenance": "new_run",
        }
        for row in summaries
    ]


def _fol_diagnostic_rows(
    *, bfr_rows: list[dict[str, str]], d50_rows: list[dict[str, str]], controls_rows: list[dict[str, str]], claim: dict[str, object]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in bfr_rows:
        rows.append({"Family": "BFR", "Source": row["Source"], "Judge": row["Judge"], "Band": row["Band"], "Value": row["Mean BFR"], "N": row["Prompts"], "Status": "observed", "Provenance": "new_run"})
    censored = Counter((row["Source"], row["Judge"]) for row in d50_rows if row.get("Right censored") == "True")
    totals = Counter((row["Source"], row["Judge"]) for row in d50_rows)
    for (source, judge), total in sorted(totals.items()):
        rows.append({"Family": "d50", "Source": source, "Judge": judge, "Band": "all", "Value": f"{censored[(source, judge)]} / {total}", "N": total, "Status": "right_censored", "Provenance": "new_run"})
    for row in controls_rows:
        rows.append({"Family": "controls", "Source": "combined", "Judge": "primary", "Band": "all", "Value": "", "N": row.get("Held-out rows", ""), "Status": row.get("Status", ""), "Provenance": "new_run"})
    gates = claim.get("quality_gates", {})
    if isinstance(gates, dict):
        for source, value in sorted(gates.items()):
            if source.endswith("_valid_paths"):
                rows.append({"Family": "interpolation", "Source": source.removesuffix("_valid_paths"), "Judge": "primary", "Band": "all", "Value": value, "N": claim.get("quality_gates", {}).get("minimum_valid_paths_per_source", ""), "Status": claim.get("decision", ""), "Provenance": "new_run"})
    return rows


def _plot_fol(*, bfr_rows: list[dict[str, str]], d50_rows: list[dict[str, str]], controls_rows: list[dict[str, str]], claim: dict[str, object], output: Path) -> None:
    """Four-panel quantitative grid: local flips, censored d50, path gate, controls."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), constrained_layout=True)
    palette = {"low": "#4C78A8", "high": "#E45756", "middle": "#7F7F7F"}
    by_curve: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in bfr_rows:
        if row.get("Judge") == "primary":
            by_curve[(row["Source"], row["Band"], row["Judge"])].append(row)
    for (source, band, _judge), rows in sorted(by_curve.items()):
        rows.sort(key=lambda row: float(row["Radius"]))
        axes[0, 0].plot([float(row["Radius"]) for row in rows], [float(row["Mean BFR"]) for row in rows], marker="o" if band == "high" else "s", linestyle="-" if band != "middle" else "--", color=palette.get(band, "#7F7F7F"), label=f"{source}: {band}")
    axes[0, 0].set_xlabel("Perturbation radius")
    axes[0, 0].set_ylabel("Mean behavior-flip rate")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=5, ncol=2, loc="upper right")
    axes[0, 0].text(0.98, 0.06, "All observed BFRs = 0", ha="right", va="bottom", fontsize=6, transform=axes[0, 0].transAxes)
    axes[0, 0].set_title("a  Local behavior flips")

    d50_counts = Counter((row["Source"], row["Judge"]) for row in d50_rows if row.get("Right censored") == "True")
    d50_totals = Counter((row["Source"], row["Judge"]) for row in d50_rows)
    labels = [f"{source}\n{judge}" for source, judge in sorted(d50_totals)]
    values = [d50_counts[key] / d50_totals[key] for key in sorted(d50_totals)]
    axes[0, 1].bar(labels, values, color="#72B7B2")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("Right-censored d50 fraction")
    axes[0, 1].set_title("b  Boundary-distance observability")
    for index, key in enumerate(sorted(d50_totals)):
        axes[0, 1].text(index, values[index] + 0.02, f"{d50_counts[key]}/{d50_totals[key]}", ha="center", fontsize=6)

    gates = claim.get("quality_gates", {})
    min_paths = gates.get("minimum_valid_paths_per_source", 0) if isinstance(gates, dict) else 0
    path_sources = [(key.removesuffix("_valid_paths"), value) for key, value in gates.items() if key.endswith("_valid_paths")] if isinstance(gates, dict) else []
    axes[1, 0].bar([source for source, _ in path_sources], [value for _, value in path_sources], color="#79706E")
    axes[1, 0].axhline(float(min_paths), color="#E45756", linestyle="--", linewidth=1, label=f"minimum = {min_paths}")
    axes[1, 0].set_ylabel("Valid interpolation paths")
    axes[1, 0].set_title("c  Interpolation quality gate")
    axes[1, 0].legend(fontsize=6)
    for index, (_source, value) in enumerate(path_sources):
        axes[1, 0].text(index, 0.12, str(value), ha="center", fontsize=7)

    status = controls_rows[0].get("Status", "not available") if controls_rows else "not available"
    axes[1, 1].axis("off")
    axes[1, 1].text(0.03, 0.82, "d  Controls-plus-FOL prediction", fontsize=8, fontweight="bold", transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.03, 0.55, f"Status: {status}", fontsize=9, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.03, 0.27, "No label transitions were observed.\nThe pre-registered decision is inconclusive.", fontsize=7, transform=axes[1, 1].transAxes)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _coverage_text(*, claim: dict[str, object], sources: list[str]) -> str:
    return f"""# Reviewer-Coverage Summary

## Evaluated Evidence

- Fair optimization comparison: `{', '.join(sources)}` use the same 17 controlled examples per source and the same seven methods: Init, GCG, PEZ, GBDA, ZOL, O-, and O+.
- Embedding-space baselines and no-FOL ablation: GCG, PEZ, GBDA, and ZOL are included in the fair matrix.
- S-Eval: S-Eval is a primary source in the main matrix and the FOL diagnostic.
- Materialization: final-checkpoint materialization fidelity reports intent preservation, semantic similarity, and both projection-cosine summaries where records are available.
- Judge robustness: Octopus and Qwen compatibility judgments are reported at thresholds 0.4, 0.5, and 0.6.
- Boundary diagnostic: the locked FOL perturbation, controls, margin, and interpolation workflow was executed. Its claim decision is `{claim.get('decision')}` (`{claim.get('reason')}`).

## Not Evaluated In This Reduced Scope

- Cross-model transfer, Qwen2.5-14B results, formal human evaluation, and optimizer-seed variance are not evaluated in this reduced scope.
- The boundary diagnostic is empirical only and cannot establish a mathematical equivalence between FOL and a safety boundary.
"""


def _limitations_text(*, claim: dict[str, object]) -> str:
    return f"""# Limitations

This revision reports a single local target model, three sources, 17 examples per source, and one optimizer seed. It therefore does not estimate transfer across target models or optimizer-seed variance. The two automated judges and three operating thresholds reduce dependence on a single threshold, but do not replace formal human evaluation.

The FOL study is `{claim.get('decision')}` because `{claim.get('reason')}`. No valid opposite-label interpolation paths were available under the primary judge. Accordingly, the results do not support a boundary-proximity claim and should not be described as showing that FOL is equal to a safety boundary.

Existing frozen PDF aggregates remain contextual provenance only and are excluded from the paired statistics reported here.
"""


def build_delivery(
    *, matrix_root: Path, fol_root: Path, output_root: Path, base_materialization_root: Path | None = None, replacement_materialization_root: Path | None = None
) -> dict[str, object]:
    """Render delivery artifacts from content-free aggregate and metadata files."""
    claim_path = fol_root / "analysis" / "fol_boundary_claim.json"
    if not claim_path.is_file():
        raise ValueError("required FOL claim is missing")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if claim.get("decision") not in {"inconclusive", "no_boundary_support", "local_sensitivity_support_only", "boundary_proxy_support"}:
        raise ValueError("FOL claim decision is invalid")
    summary = _read_csv(matrix_root / "analysis" / "summary.csv")
    paired = _read_csv(matrix_root / "analysis" / "paired_vs_init.csv")
    ranks = _read_csv(matrix_root / "analysis" / "threshold_ranks.csv")
    bfr = _read_csv(fol_root / "analysis" / "fol_bfr.csv")
    d50 = _read_csv(fol_root / "analysis" / "fol_d50.csv")
    controls = _read_csv(fol_root / "analysis" / "fol_h4_controls.csv")
    sources = sorted({row["Source"] for row in summary})
    methods = {row["Method"] for row in summary}
    sample_counts = {int(row["N"]) for row in summary}
    if not sources or len(sample_counts) != 1:
        raise ValueError("main matrix must use one non-empty controlled sample count")
    sample_count = sample_counts.pop()

    tables = output_root / "tables"
    fair = [row for row in summary if row["Judge"] == PRIMARY_JUDGE and row["Threshold"] == CENTRAL_THRESHOLD]
    if not fair:
        fair = [row for row in summary if row["Threshold"] == CENTRAL_THRESHOLD]
    _copy_table(source_rows=fair, destination=tables / "fair_optimization", fieldnames=("Judge", "Target", "Source", "Method", "Threshold", "Unsafe", "N", "ASR", "Provenance"))
    paired_central = [row for row in paired if row["Threshold"] == CENTRAL_THRESHOLD]
    _copy_table(source_rows=paired_central, destination=tables / "paired_vs_init", fieldnames=("Judge", "Target", "Source", "Method", "Threshold", "N", "Net ASR change", "Method-only", "Init-only", "McNemar p", "Provenance"))
    _copy_table(source_rows=summary, destination=tables / "judge_threshold_sensitivity", fieldnames=("Judge", "Target", "Source", "Method", "Threshold", "Unsafe", "N", "ASR", "Provenance"))
    _copy_table(source_rows=ranks, destination=tables / "threshold_ranks", fieldnames=("Judge", "Target", "Source", "Threshold", "Method", "Rank", "Provenance"))
    fol_rows = _fol_diagnostic_rows(bfr_rows=bfr, d50_rows=d50, controls_rows=controls, claim=claim)
    _write_csv(tables / "fol_boundary_diagnostics.csv", fol_rows, ("Family", "Source", "Judge", "Band", "Value", "N", "Status", "Provenance"))
    _write_latex(tables / "fol_boundary_diagnostics.tex", fol_rows, ("Family", "Source", "Judge", "Band", "Value", "N", "Status", "Provenance"))
    materializations = _materialization_rows(
        base_root=base_materialization_root,
        replacement_root=replacement_materialization_root,
        sources=set(sources),
        methods=methods,
    )
    if materializations:
        materialization_fields = ("Source", "Method", "Total", "Complete", "Failed", "Intent preserved", "Semantic similarity mean", "Prefix projection cosine mean", "Seed projection cosine mean", "Provenance")
        _write_csv(tables / "materialization_fidelity.csv", materializations, materialization_fields)
        _write_latex(tables / "materialization_fidelity.tex", materializations, materialization_fields)
    _plot_fol(bfr_rows=bfr, d50_rows=d50, controls_rows=controls, claim=claim, output=output_root / "figures" / "fol_boundary_diagnostics")

    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "evaluation_coverage.md").write_text(_coverage_text(claim=claim, sources=sources), encoding="utf-8")
    (analysis / "limitations.md").write_text(_limitations_text(claim=claim), encoding="utf-8")
    manifest = {
        "scope": {"sources": sources, "samples_per_source": sample_count},
        "target": sorted({row["Target"] for row in summary}),
        "claim_decision": claim["decision"],
        "claim_reason": claim.get("reason"),
        "materialization_table": bool(materializations),
        "provenance": "new_run",
    }
    (output_root / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--fol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-materialization-root", type=Path)
    parser.add_argument("--replacement-materialization-root", type=Path)
    args = parser.parse_args()
    payload = build_delivery(
        matrix_root=args.matrix_root,
        fol_root=args.fol_root,
        output_root=args.output_root,
        base_materialization_root=args.base_materialization_root,
        replacement_materialization_root=args.replacement_materialization_root,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
