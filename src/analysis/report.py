"""Result aggregation and Markdown/LaTeX report generation for the benchmark suite."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from metrics.benchmark_metrics import compute_all_metrics, ALL_THREAT_CATEGORIES
from judge.llm_judge import LLMJudge

logger = logging.getLogger(__name__)

# Human-readable method display names, keyed by filename stem prefix
METHOD_DISPLAY_NAMES: dict[str, str] = {
    "pair": "B0 — PAIR Baseline",
    "tap": "B1 — TAP Baseline",
    "gcg": "B2 — GCG/REINFORCE Baseline",
    "ablation_gpo": "A1 — GPO Only",
    "ablation_unalign": "A2 — Unalignment Model Only",
    "ablation_gpo_unalign": "A3 — GPO + Unalignment",
    "ablation_full": "A4 — Full Framework",
    "cross_model": "T1 — Cross-Model Transfer",
    "cross_attack": "T2 — Cross-Attack Transfer",
    "cross_threat": "T3 — Cross-Threat Transfer",
}


def _stem(filename: str) -> str:
    """Return the filename stem without extension."""
    return os.path.splitext(os.path.basename(filename))[0]


def _method_display(stem: str) -> str:
    """Resolve a display name for a given result file stem."""
    for key, display in METHOD_DISPLAY_NAMES.items():
        if stem.startswith(key) or stem.endswith(key):
            return display
    return stem


def load_results(results_dir: str) -> dict[str, list[dict]]:
    """Load all JSON/JSONL result files from a directory tree.

    Recursively walks *results_dir* and collects every *.json / *.jsonl file.
    JSONL files are read line-by-line; JSON files may be a list or a dict.

    Args:
        results_dir: Root directory containing result files (may have subdirs).

    Returns:
        Dict mapping experiment name (file stem) → list of result dicts.
    """
    all_results: dict[str, list[dict]] = {}

    for root, _dirs, files in os.walk(results_dir):
        for fname in sorted(files):
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            fpath = os.path.join(root, fname)
            stem = _stem(fname)
            records: list[dict] = []
            try:
                if fname.endswith(".jsonl"):
                    with open(fpath, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                records.append(json.loads(line))
                else:
                    with open(fpath, encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        records = data
                    elif isinstance(data, dict):
                        # Might be a summary dict — wrap or unwrap results key
                        records = data.get("results", [data])
            except Exception as exc:
                logger.warning("Could not load %s: %s", fpath, exc)
                continue

            if records:
                all_results[stem] = records

    return all_results


def generate_markdown_table(all_metrics: dict[str, dict]) -> str:
    """Generate a Markdown results table from pre-computed metric dicts.

    Args:
        all_metrics: Dict mapping experiment name → metrics dict
                     (keys: asr, js, coverage, n_total, n_success).

    Returns:
        Formatted Markdown table string.
    """
    header = "| Method | N | ASR | JS | Coverage |"
    separator = "|--------|---|-----|----|---------| "
    rows: list[str] = [header, separator]

    for stem, metrics in sorted(all_metrics.items()):
        display = _method_display(stem)
        n = metrics.get("n_total", 0)
        asr = metrics.get("asr", 0.0)
        js = metrics.get("js", 0.0)
        cov = metrics.get("coverage", 0.0)
        n_success = metrics.get("n_success", 0)
        cov_frac = (
            f"{round(cov * len(ALL_THREAT_CATEGORIES))}/{len(ALL_THREAT_CATEGORIES)}"
        )
        row = f"| {display} | {n} ({n_success} ✓) | {asr:.2%} | {js:.2f} | {cov_frac} |"
        rows.append(row)

    return "\n".join(rows)


def generate_report(
    results_dir: str,
    output_file: str = "report.md",
    judge: LLMJudge | None = None,
) -> None:
    """Load all experiment results, compute metrics, and write a Markdown report.

    Args:
        results_dir: Root directory containing result subdirectories.
        output_file: Path for the output Markdown report.
        judge: Optional LLMJudge for re-scoring; skipped if None.
    """
    all_results = load_results(results_dir)
    if not all_results:
        logger.warning("No result files found in %s", results_dir)

    all_metrics: dict[str, dict] = {}
    for stem, records in all_results.items():
        try:
            metrics = compute_all_metrics(records, judge=judge)
        except Exception as exc:
            logger.warning("Metric computation failed for %s: %s", stem, exc)
            metrics = {
                "asr": 0.0,
                "js": 0.0,
                "coverage": 0.0,
                "n_total": 0,
                "n_success": 0,
            }
        all_metrics[stem] = metrics

    table = generate_markdown_table(all_metrics)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# Benchmark Report",
        "",
        f"Generated: {timestamp}",
        f"Results directory: `{results_dir}`",
        f"Experiments loaded: {len(all_results)}",
        "",
        "## Results Summary",
        "",
        table,
        "",
        "## Per-Experiment Details",
        "",
    ]

    for stem, metrics in sorted(all_metrics.items()):
        display = _method_display(stem)
        lines += [
            f"### {display}",
            "",
            f"- **N total**: {metrics['n_total']}",
            f"- **N success**: {metrics['n_success']}",
            f"- **ASR**: {metrics['asr']:.4f}",
            f"- **JS (mean judge score)**: {metrics['js']:.4f}",
            f"- **Coverage**: {metrics['coverage']:.4f}",
            "",
        ]

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    print(f"Report written to: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown benchmark report from experiment results."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory containing result JSON/JSONL files (default: results/).",
    )
    parser.add_argument(
        "--output",
        default="results/report.md",
        help="Output path for the Markdown report (default: results/report.md).",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-72b",
        help="Judge model name (default: qwen-72b).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BENCHMARK_API_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL (or BENCHMARK_API_BASE_URL).",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge; use scores already stored in result files.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()

    judge_instance: LLMJudge | None = None
    if not args.no_judge:
        judge_instance = LLMJudge(model=args.judge_model, base_url=args.base_url)

    generate_report(
        results_dir=args.results_dir,
        output_file=args.output,
        judge=judge_instance,
    )
