"""Run the approved three-source safety-evaluation matrix with one fail-closed gate per method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.io import atomic_write_json, canonical_hash
from benchmark.safety_eval.judging import Qwen32CompatJudge, thresholds
from benchmark.safety_eval.method_gated import MethodGateError, run_method_gate


SOURCES = ("harmbench", "jailbound", "s_eval")
METHODS = ("init", "gcg", "pez", "gbda", "zol", "jailbound_o_minus", "jailbound_o_plus")


def _selection_hash(*, parent_root: Path, target_key: str, samples_per_source: int) -> str:
    return canonical_hash(
        {
            "methods": METHODS,
            "parent_root": str(parent_root.resolve()),
            "samples_per_source": samples_per_source,
            "sources": SOURCES,
            "target_key": target_key,
        }
    )


def _write_selection_manifest(
    *,
    selection_root: Path,
    parent_root: Path,
    target_key: str,
    samples_per_source: int,
    selection_hash: str,
) -> None:
    atomic_write_json(
        selection_root / "selection_manifest.json",
        {
            "methods": list(METHODS),
            "parent_root": str(parent_root.resolve()),
            "samples_per_source": samples_per_source,
            "selection_hash": selection_hash,
            "sources": list(SOURCES),
            "target_key": target_key,
        },
    )


def run_selected_methods(
    *,
    parent_root: Path,
    selection_root: Path,
    target_key: str,
    samples_per_source: int,
    primary_key: str,
    thresholds: tuple[float, ...],
    judge: Any,
    selection_hash: str,
) -> list[dict[str, object]]:
    """Run every approved method serially and stop at the first failed gate."""
    _write_selection_manifest(
        selection_root=selection_root,
        parent_root=parent_root,
        target_key=target_key,
        samples_per_source=samples_per_source,
        selection_hash=selection_hash,
    )
    results: list[dict[str, object]] = []
    for method in METHODS:
        evidence = run_method_gate(
            parent_root=parent_root,
            selection_root=selection_root,
            sources=SOURCES,
            method=method,
            target_key=target_key,
            samples_per_source=samples_per_source,
            primary_key=primary_key,
            thresholds=thresholds,
            judge=judge,
            selection_hash=selection_hash,
        )
        results.append(
            {
                "method": evidence.method,
                "primary_judgment_count": evidence.primary_judgment_count,
                "response_count": evidence.response_count,
                "secondary_judgment_count": evidence.secondary_judgment_count,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.target not in {target.key for target in config.models.targets}:
        raise ValueError("target is not configured")
    if set(SOURCES) - set(config.data.sources) or set(METHODS) - set(config.optimization.methods):
        raise ValueError("approved selected matrix is not present in the parent configuration")
    secondary = config.judging.secondary
    if secondary.endpoint is None or secondary.model is None:
        raise ValueError("secondary compatibility endpoint is required")
    selected_thresholds = tuple(thresholds(secondary.threshold, secondary.threshold_offsets))
    selection_hash = _selection_hash(
        parent_root=args.parent_root,
        target_key=args.target,
        samples_per_source=config.data.samples_per_source,
    )
    try:
        with Qwen32CompatJudge(
            endpoint=secondary.endpoint,
            model=secondary.model,
            max_new_tokens=config.judging.max_new_tokens,
        ) as judge:
            result = run_selected_methods(
                parent_root=args.parent_root,
                selection_root=args.selection_root,
                target_key=args.target,
                samples_per_source=config.data.samples_per_source,
                primary_key=config.judging.primary.key,
                thresholds=selected_thresholds,
                judge=judge,
                selection_hash=selection_hash,
            )
    except MethodGateError as error:
        print(json.dumps({"failure": str(error), "status": "failed"}, sort_keys=True))
        return 1
    print(json.dumps({"methods": result, "status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
