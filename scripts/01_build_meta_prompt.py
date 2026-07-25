#!/usr/bin/env python3
"""Step 01: Build meta-attack-prompts.

Given threat category × attack type, drive the unalignment model M_a to emit
structured meta-attack-prompts (see paper §3.1).

Wires: taxonomy + generator.meta_prompt + generator.unalignment
Output: data/meta_attack/*.jsonl

Usage:
    # Single prompt
    python scripts/01_build_meta_prompt.py \
        --threat cybersecurity_misuse --attack scenario_nesting \
        --intent "Explain how to exploit a buffer overflow"

    # Batch from JSONL (each line: {"threat_category": ..., "attack_type": ..., "malicious_intent": ...})
    python scripts/01_build_meta_prompt.py \
        --input data/intents.jsonl --output data/meta_attack/batch_output.jsonl

    # Use vLLM endpoint for generation
    python scripts/01_build_meta_prompt.py \
        --threat cybersecurity_misuse --attack scenario_nesting \
        --intent "..." --use-vllm --model qwen-72b
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator.meta_prompt.builder import MetaPromptBuilder  # noqa: E402
from taxonomy.threat_taxonomy import THREAT_CATEGORIES  # noqa: E402
from taxonomy.attack_types import ATTACK_TYPES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build meta-attack-prompts via MetaPromptBuilder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--threat", type=str, default=None,
                   help=f"Threat category key. Options: {list(THREAT_CATEGORIES.keys())}")
    p.add_argument("--attack", type=str, default=None,
                   help=f"Attack type key. Options: {list(ATTACK_TYPES.keys())}")
    p.add_argument("--intent", type=str, default=None,
                   help="Malicious intent description (single mode).")
    p.add_argument("--input", type=Path, default=None,
                   help="Input JSONL for batch mode (fields: threat_category, attack_type, malicious_intent).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSONL path. Defaults to stdout for single mode.")
    p.add_argument("--use-vllm", action="store_true",
                   help="Use vLLM endpoint to generate attack_prompt text.")
    p.add_argument("--model", type=str, default="qwen-72b",
                   help="vLLM model name (default: qwen-72b).")
    p.add_argument("--base-url", type=str, default=os.environ.get("BENCHMARK_API_BASE_URL", "http://localhost:8000"),
                   help="vLLM base URL.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    builder = MetaPromptBuilder(
        use_vllm=args.use_vllm,
        vllm_model=args.model,
        vllm_base_url=args.base_url,
    )

    # --- Batch mode ---
    if args.input:
        if not args.input.exists():
            parser.error(f"Input file not found: {args.input}")
        if not args.output:
            parser.error("--output is required in batch mode.")

        requests = []
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    requests.append(json.loads(line))

        logger.info("Batch mode: %d requests from %s", len(requests), args.input)
        results = builder.build_batch(requests)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as out:
            for meta in results:
                out.write(json.dumps(meta.__dict__, ensure_ascii=False) + "\n")
        logger.info("Wrote %d meta-attack-prompts to %s", len(results), args.output)
        return

    # --- Single mode ---
    if not args.threat or not args.attack or not args.intent:
        parser.error("Single mode requires --threat, --attack, and --intent.")

    meta = builder.build(
        threat_category=args.threat,
        attack_type=args.attack,
        malicious_intent=args.intent,
    )

    record = json.dumps(meta.__dict__, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as out:
            out.write(record + "\n")
        logger.info("Wrote to %s", args.output)
    else:
        print(record)


if __name__ == "__main__":
    main()
