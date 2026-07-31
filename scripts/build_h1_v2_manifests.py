"""Freeze new, disjoint 81-candidate manifests for the H1-v2 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark.safety_eval.config import load_h1_v2_config
from benchmark.safety_eval.datasets import load_source_with_report
from benchmark.safety_eval.fol_runtime import select_h1_v2_candidates
from benchmark.safety_eval.io import atomic_write_json, read_jsonl, sha256_file
from benchmark.safety_eval.manifest import map_raw_candidates, write_controlled_manifest
from benchmark.safety_eval.runtime import lock_runtime_config
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder, load_taxonomy_mapping


ROOT = Path(__file__).resolve().parents[1]


def _prior_ids(root: Path, source: str) -> set[str]:
    """Read identities only; existing prompt content is never loaded or emitted."""
    path = root / "manifests" / f"controlled_{source}.jsonl"
    return {
        value
        for row in read_jsonl(path)
        for value in (row.get("example_id"),)
        if isinstance(value, str) and value
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=240)
    args = parser.parse_args()
    config = load_h1_v2_config(args.config)
    if args.candidate_pool < config.h1_v2.candidate_count:
        raise ValueError("H1-v2 candidate pool must contain at least 81 rows")
    root = ROOT / config.h1_v2.output_root
    base_root = ROOT / config.base.run.output_root
    source_hashes = {
        source: sha256_file(ROOT / config.base.data.paths[source])
        for source in config.h1_v2.sources
    }
    locked = lock_runtime_config(config, output_root=root, source_hashes=source_hashes)
    model_path = config.base.models.semantic_encoder.local_path
    if model_path is None:
        raise ValueError("H1-v2 manifest construction requires a local semantic encoder")
    mapping = load_taxonomy_mapping(ROOT / "configs/benchmark/safety_eval_taxonomy_map.yaml")
    labels = list(mapping["risk_categories"]) + list(mapping["threat_domains"])
    descriptions = [
        mapping["risk_categories"].get(label, mapping["threat_domains"].get(label))["description"]
        for label in labels
    ]
    encoder = QwenHiddenMeanEncoder(model_path)
    label_embeddings = dict(zip(labels, encoder.encode(descriptions), strict=True))
    reports: dict[str, object] = {}
    for source in config.h1_v2.sources:
        raw, report = load_source_with_report(
            source,
            ROOT / config.base.data.paths[source],
            ROOT / config.base.data.harmbench_targets_path if source == "harmbench" else None,
        )
        candidate_pool = sorted(
            raw,
            key=lambda row: hashlib.sha256(
                f"{config.base.run.seed}|h1-v2|{row.source_row_id}".encode()
            ).hexdigest(),
        )[: args.candidate_pool]
        mapped = map_raw_candidates(
            candidate_pool,
            mapping=mapping,
            label_embeddings=label_embeddings,
            encoder=encoder,
            source_file=str(config.base.data.paths[source]),
            source_sha256=source_hashes[source],
            seed=config.base.run.seed,
        )
        main_ids = _prior_ids(base_root, source)
        exploratory_ids = _prior_ids(base_root / "fol_boundary", source)
        selected = select_h1_v2_candidates(
            mapped,
            excluded_ids=main_ids | exploratory_ids,
            seed=config.base.run.seed,
        )
        header = write_controlled_manifest(
            root,
            source,
            selected,
            source_file_sha256=source_hashes[source],
            config_hash=locked.config_hash,
        )
        reports[source] = {
            "candidate_ids": [row.example_id for row in selected],
            "candidate_pool": len(candidate_pool),
            "eligible_count": report.eligible_count,
            "main_matrix_excluded": len(main_ids),
            "exploratory_fol_excluded": len(exploratory_ids),
            "manifest_hash": header.manifest_hash,
            "record_count": header.record_count,
        }
    atomic_write_json(root / "manifests" / "h1_v2_candidate_selection.json", {
        "config_hash": locked.config_hash,
        "encoder_revision": encoder.resolved_revision,
        "sources": reports,
    })
    print(json.dumps({"sources": list(config.h1_v2.sources), "status": "frozen"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
