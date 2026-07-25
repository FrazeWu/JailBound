#!/usr/bin/env python3
"""Step 03: Optimize meta-attack-prompts into optimal-attack-prompts.

Runs the QuoTe v2 FOL-guided bi-end search over seed prompts and writes the
selected, distilled attack prompts as JSONL.

Input JSONL accepts records from Step 01 or explicit seed records. The loader maps:
    meta_prompt: attack_prompt | meta_prompt | system_prompt
    original_seed: malicious_intent | original_seed | behavior | prompt
    condition_id: condition_id | <threat_category>:<attack_type>
    behavior_id: id | behavior_id | row index
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.quote_config import QuoTeConfig  # noqa: E402
from materialization.distillation import distill_to_text  # noqa: E402
from materialization.model_loader import load_model  # noqa: E402
from optimizer.biend_search import run_biend_search  # noqa: E402
from optimizer.candidate_selection import select_optimal_attacks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_seeds(path: Path) -> list[dict[str, str]]:
    seeds: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta_prompt = row.get("attack_prompt") or row.get("meta_prompt") or row.get("system_prompt")
            original_seed = (
                row.get("malicious_intent")
                or row.get("original_seed")
                or row.get("behavior")
                or row.get("prompt")
            )
            if not meta_prompt or not original_seed:
                raise ValueError(
                    f"{path}:{i + 1}: expected meta prompt and seed fields; got keys={list(row.keys())}"
                )
            threat = row.get("threat_category", "")
            attack = row.get("attack_type", "")
            seeds.append(
                {
                    "meta_prompt": str(meta_prompt),
                    "original_seed": str(original_seed),
                    "condition_id": str(row.get("condition_id") or f"{threat}:{attack}"),
                    "behavior_id": str(row.get("behavior_id") or row.get("id") or f"seed_{i}"),
                }
            )
    return seeds


def _state_record(state: Any, loaded: Any) -> dict[str, Any]:
    return {
        "behavior_id": state.behavior_id,
        "condition_id": state.condition_id,
        "branch_type": state.branch_type,
        "step": state.step,
        "meta_prompt": state.meta_prompt,
        "original_seed": state.original_seed,
        "mutated_seed": state.mutated_seed,
        "optimal_attack_prompt": distill_to_text(state, loaded),
        "risk_score": state.risk_score,
        "proxy_risk": state.proxy_risk,
        "zol": state.zol,
        "fol": state.fol,
        "prefix_norm": state.prefix_norm,
        "seed_block_drift": state.seed_block_drift,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optimize meta-attack-prompts with QuoTe v2.")
    p.add_argument("--input", type=Path, required=True, help="Input meta prompt / seed JSONL.")
    p.add_argument("--output", type=Path, required=True, help="Output optimized attack JSONL.")
    p.add_argument("--model-path", default=None, help="Local/HF model path for the white-box surrogate.")
    p.add_argument("--device", default=None, help="Torch device (default from QuoTeConfig).")
    p.add_argument("--torch-dtype", default=None, help="float32 | float16 | bfloat16.")
    p.add_argument("--epsilon", type=float, default=None, help="Override QuoTeConfig.epsilon.")
    p.add_argument("--eta", type=float, default=None, help="Override QuoTeConfig.eta.")
    p.add_argument("--max-opt-steps", type=int, default=None, help="Override QuoTeConfig.max_opt_steps.")
    p.add_argument("--select-top-k", type=int, default=None, help="Override QuoTeConfig.select_top_k.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = QuoTeConfig()

    if args.model_path is not None:
        config.model_path = args.model_path
    if args.device is not None:
        config.device = args.device
    if args.torch_dtype is not None:
        config.torch_dtype = args.torch_dtype
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.eta is not None:
        config.eta = args.eta
    if args.max_opt_steps is not None:
        config.max_opt_steps = args.max_opt_steps
    if args.select_top_k is not None:
        config.select_top_k = args.select_top_k

    seeds = _load_seeds(args.input)
    logger.info("Loaded %d seeds from %s", len(seeds), args.input)

    loaded = load_model(
        config.model_path,
        device=config.device,
        torch_dtype=config.torch_dtype,
    )
    pools, diagnostics = run_biend_search(seeds, loaded, config)
    selected = select_optimal_attacks(pools, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for state in selected:
            f.write(json.dumps(_state_record(state, loaded), ensure_ascii=False) + "\n")

    diag_path = args.output.with_suffix(args.output.suffix + ".diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d optimized prompts to %s", len(selected), args.output)
    logger.info("Wrote diagnostics to %s", diag_path)


if __name__ == "__main__":
    main()
