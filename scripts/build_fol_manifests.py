"""Build disjoint 45-example FOL candidate manifests for JailBound and S-Eval."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from benchmark.safety_eval.config import load_config
from benchmark.safety_eval.datasets import load_source_with_report
from benchmark.safety_eval.fol_runtime import select_fol_candidates
from benchmark.safety_eval.io import atomic_write_json, read_jsonl, sha256_file
from benchmark.safety_eval.manifest import (
    map_raw_candidates,
    write_controlled_manifest,
)
from benchmark.safety_eval.runtime import lock_runtime_config
from benchmark.safety_eval.semantic import QwenHiddenMeanEncoder, load_taxonomy_mapping


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=200)
    args = parser.parse_args()
    if args.candidate_pool < 45:
        raise ValueError("FOL candidate pool must contain at least 45 rows")

    config = load_config(args.config)
    output_root = ROOT / config.run.output_root
    fol_root = output_root / "fol_boundary"
    source_hashes = {
        source: sha256_file(ROOT / config.data.paths[source])
        for source in config.fol.sources
    }
    locked = lock_runtime_config(config, output_root=fol_root, source_hashes=source_hashes)
    model_path = config.models.semantic_encoder.local_path
    if model_path is None:
        raise ValueError("FOL manifest construction requires a local semantic encoder")
    mapping = load_taxonomy_mapping(ROOT / "configs/benchmark/safety_eval_taxonomy_map.yaml")
    labels = list(mapping["risk_categories"]) + list(mapping["threat_domains"])
    descriptions = [
        mapping["risk_categories"].get(label, mapping["threat_domains"].get(label))["description"]
        for label in labels
    ]
    encoder = QwenHiddenMeanEncoder(model_path)
    label_embeddings = dict(zip(labels, encoder.encode(descriptions), strict=True))
    reports: dict[str, object] = {}
    for source in config.fol.sources:
        raw, report = load_source_with_report(
            source,
            ROOT / config.data.paths[source],
            ROOT / config.data.harmbench_targets_path if source == "harmbench" else None,
        )
        chosen = sorted(
            raw,
            key=lambda row: hashlib.sha256(
                f"{config.run.seed}|fol|{row.source_row_id}".encode()
            ).hexdigest(),
        )[: args.candidate_pool]
        mapped = map_raw_candidates(
            chosen,
            mapping=mapping,
            label_embeddings=label_embeddings,
            encoder=encoder,
            source_file=str(config.data.paths[source]),
            source_sha256=source_hashes[source],
            seed=config.run.seed,
        )
        main_ids = {
            str(row["example_id"])
            for row in read_jsonl(output_root / "manifests" / f"controlled_{source}.jsonl")
        }
        selected = select_fol_candidates(mapped, excluded_ids=main_ids, seed=config.run.seed)
        header = write_controlled_manifest(
            fol_root,
            source,
            selected,
            source_file_sha256=source_hashes[source],
            config_hash=locked.config_hash,
        )
        reports[source] = {
            "candidate_pool": len(chosen),
            "eligible_count": report.eligible_count,
            "main_matrix_excluded": len(main_ids),
            "manifest_hash": header.manifest_hash,
            "record_count": header.record_count,
        }
    atomic_write_json(
        fol_root / "manifests" / "candidate_selection.json",
        {
            "config_hash": locked.config_hash,
            "encoder_revision": encoder.resolved_revision,
            "reports": reports,
        },
    )
    print({"sources": list(config.fol.sources), "status": "frozen"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
