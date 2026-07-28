"""Build auditable controlled manifests with the local-Qwen compatibility encoder."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.datasets import load_source_with_report
from benchmark.safety_eval.io import atomic_write_json, canonical_hash, sha256_file
from benchmark.safety_eval.manifest import build_controlled_manifests, map_raw_candidates
from benchmark.safety_eval.runtime import lock_runtime_config
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder, load_taxonomy_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=200)
    args = parser.parse_args()
    root = args.config.resolve().parents[2]
    config = load_config(args.config)
    source_hashes = {name: sha256_file(root / path) for name, path in config.data.paths.items()}
    locked = lock_runtime_config(config, output_root=root / config.run.output_root, source_hashes=source_hashes)
    mapping = load_taxonomy_mapping(root / "configs/benchmark/safety_eval_taxonomy_map.yaml")
    labels = list(mapping["risk_categories"]) + list(mapping["threat_domains"])
    descriptions = [mapping["risk_categories"].get(label, mapping["threat_domains"].get(label))["description"] for label in labels]
    encoder = QwenHiddenMeanEncoder(config.models.semantic_encoder.local_path)
    vectors = encoder.encode(descriptions)
    embeddings = dict(zip(labels, vectors, strict=True))
    candidates = {}
    reports = {}
    for source in config.data.sources:
        raw, report = load_source_with_report(source, root / config.data.paths[source], root / config.data.harmbench_targets_path if source == "harmbench" else None)
        chosen = sorted(raw, key=lambda row: hashlib.sha256(f"{config.run.seed}|{row.source_row_id}".encode()).hexdigest())[:args.candidate_pool]
        candidates[source] = map_raw_candidates(chosen, mapping=mapping, label_embeddings=embeddings, encoder=encoder, source_file=str(config.data.paths[source]), source_sha256=source_hashes[source], seed=config.run.seed)
        reports[source] = {"raw_count": report.raw_count, "eligible_count": report.eligible_count, "exclusions": report.exclusions, "candidate_pool": len(chosen)}
    headers = build_controlled_manifests(
        candidates,
        output_root=root / config.run.output_root,
        source_hashes=source_hashes,
        config_hash=locked.config_hash,
        seed=config.run.seed,
        samples_per_source=config.data.samples_per_source,
    )
    atomic_write_json(root / config.run.output_root / "manifests" / "source_ingestion_report.json", {"encoder": encoder.resolved_revision, "config_hash": locked.config_hash, "reports": reports, "manifest_hashes": {key: value.manifest_hash for key, value in headers.items()}})


if __name__ == "__main__":
    main()
