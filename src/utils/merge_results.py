"""
Merge experiment results and compute aggregate RQ1-RQ4 metrics.

Merges results from:
1. formal110 experiment (100 seeds × 3 Qwen2.5 targets)
2. incremental10 experiment (10 extra seeds × 3 Qwen2.5 targets)
3. cross-model evaluation (N seeds × all target models)

Usage:
    python merge_results.py \
        --formal outputs/formal110_XXXX/all_results.jsonl \
        --incremental outputs/incremental10_XXXX/all_results.jsonl \
        --cross_model outputs/cross_model_XXXX/cross_model_results.jsonl \
        --output_dir outputs/merged_XXXX
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--formal", type=str, nargs="+", help="Formal experiment JSONL files")
    p.add_argument("--incremental", type=str, nargs="*", default=[], help="Incremental JSONL files")
    p.add_argument("--cross_model", type=str, nargs="*", default=[], help="Cross-model JSONL files")
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_rows = []
    for f in (args.formal or []):
        rows = load_jsonl(Path(f))
        logger.info("Loaded %d rows from formal: %s", len(rows), f)
        all_rows.extend(rows)

    for f in args.incremental:
        rows = load_jsonl(Path(f))
        logger.info("Loaded %d rows from incremental: %s", len(rows), f)
        all_rows.extend(rows)

    for f in args.cross_model:
        rows = load_jsonl(Path(f))
        logger.info("Loaded %d rows from cross-model: %s", len(rows), f)
        all_rows.extend(rows)

    logger.info("Total merged: %d rows", len(all_rows))

    # Deduplicate by (variant, target_model, behavior_id)
    seen = set()
    deduped = []
    for r in all_rows:
        key = (r["variant"], r["target_model"], r["behavior_id"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    logger.info("After dedup: %d rows (removed %d duplicates)",
                len(deduped), len(all_rows) - len(deduped))

    # Save merged
    merged_path = output_dir / "all_results_merged.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # === RQ1: Main comparison table ===
    print("\n" + "=" * 80)
    print("RQ1: Main Comparison Table (ASR by Variant)")
    print("=" * 80)
    by_variant = defaultdict(lambda: {"n": 0, "success": 0, "scores": []})
    for r in deduped:
        v = r["variant"]
        by_variant[v]["n"] += 1
        by_variant[v]["scores"].append(r["judge_score"])
        if r["success"]:
            by_variant[v]["success"] += 1

    print(f"{'Variant':<35} {'N':>5} {'Success':>8} {'ASR':>8} {'Avg Score':>10}")
    print("-" * 70)
    for v in sorted(by_variant):
        d = by_variant[v]
        asr = d["success"] / max(1, d["n"])
        avg = sum(d["scores"]) / max(1, len(d["scores"]))
        print(f"{v:<35} {d['n']:>5} {d['success']:>8} {asr:>8.4f} {avg:>10.4f}")

    # === RQ2: Transfer heatmap (variant × target model) ===
    print("\n" + "=" * 80)
    print("RQ2: Transfer Heatmap (Variant × Target Model ASR)")
    print("=" * 80)
    vtm = defaultdict(lambda: {"n": 0, "success": 0})
    for r in deduped:
        key = (r["variant"], r["target_model"])
        vtm[key]["n"] += 1
        if r["success"]:
            vtm[key]["success"] += 1

    models = sorted(set(r["target_model"] for r in deduped))
    variants = sorted(set(r["variant"] for r in deduped))

    header = f"{'Variant':<35}" + "".join(f"{m:>22}" for m in models)
    print(header)
    print("-" * len(header))
    for v in variants:
        row = f"{v:<35}"
        for m in models:
            d = vtm.get((v, m), {"n": 0, "success": 0})
            if d["n"] > 0:
                asr = d["success"] / d["n"]
                row += f"{asr:>18.4f}({d['n']:>3})"
            else:
                row += f"{'—':>22}"
        print(row)

    # === RQ3: Domain × model heatmap ===
    print("\n" + "=" * 80)
    print("RQ3: Domain × Target Model ASR")
    print("=" * 80)
    # Only QuoTe variants
    quote_rows = [r for r in deduped if r["attack_source"] == "quote"]
    dm = defaultdict(lambda: {"n": 0, "success": 0})
    for r in quote_rows:
        key = (r.get("domain", "general"), r["target_model"])
        dm[key]["n"] += 1
        if r["success"]:
            dm[key]["success"] += 1

    quote_models = sorted(set(r["target_model"] for r in quote_rows))
    domains = sorted(set(r.get("domain", "general") for r in quote_rows))

    header = f"{'Domain':<20}" + "".join(f"{m:>22}" for m in quote_models)
    print(header)
    print("-" * len(header))
    for d in domains:
        row = f"{d:<20}"
        for m in quote_models:
            dd = dm.get((d, m), {"n": 0, "success": 0})
            if dd["n"] > 0:
                asr = dd["success"] / dd["n"]
                row += f"{asr:>18.4f}({dd['n']:>3})"
            else:
                row += f"{'—':>22}"
        print(row)

    # === RQ3 extra: Risk Category × model ===
    print("\n" + "=" * 80)
    print("RQ3b: Risk Category × Target Model ASR (QuoTe only)")
    print("=" * 80)
    cm = defaultdict(lambda: {"n": 0, "success": 0})
    for r in quote_rows:
        key = (r.get("risk_category", "unknown"), r["target_model"])
        cm[key]["n"] += 1
        if r["success"]:
            cm[key]["success"] += 1

    categories = sorted(set(r.get("risk_category", "unknown") for r in quote_rows))
    header = f"{'Category':<30}" + "".join(f"{m:>22}" for m in quote_models)
    print(header)
    print("-" * len(header))
    for c in categories:
        row = f"{c:<30}"
        for m in quote_models:
            dd = cm.get((c, m), {"n": 0, "success": 0})
            if dd["n"] > 0:
                asr = dd["success"] / dd["n"]
                row += f"{asr:>18.4f}({dd['n']:>3})"
            else:
                row += f"{'—':>22}"
        print(row)

    # Save summary JSON
    summary = {
        "total_rows": len(deduped),
        "total_success": sum(1 for r in deduped if r["success"]),
        "overall_asr": round(sum(1 for r in deduped if r["success"]) / max(1, len(deduped)), 4),
        "rq1": {v: {"n": d["n"], "success": d["success"],
                     "asr": round(d["success"] / max(1, d["n"]), 4)}
                for v, d in by_variant.items()},
        "rq2": {f"{v}|{m}": {"n": d["n"], "asr": round(d["success"] / max(1, d["n"]), 4)}
                for (v, m), d in vtm.items()},
        "models_evaluated": models,
        "variants": variants,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved merged results to %s", merged_path)
    logger.info("Saved summary to %s", output_dir / "summary.json")


if __name__ == "__main__":
    main()
