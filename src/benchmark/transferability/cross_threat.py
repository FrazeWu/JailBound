"""T3 Transferability — Cross-threat-category.

Tests the A4 full framework across all 12 threat categories.  For each
category, only behaviors belonging to that category are used to measure
ASR and coverage within that threat domain.

Threat categories (12):
    discrimination_toxicity, sexual_graphic, privacy_personal_data,
    sensitive_org_gov, cybersecurity_misuse, illegal_criminal,
    fraud_scam, malicious_influence, misinformation_reliability,
    high_stakes_advice, unsafe_unethical, human_chatbot_harm

CLI usage:
    python cross_threat.py \\
        --behaviors data/harmbench_behaviors.jsonl \\
        --output results/transferability/cross_threat.json \\
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from utils.bench_utils import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_openai_client,
    call_target_model,
    load_behaviors,
    save_jsonl,
)
from judge.llm_judge import LLMJudge
from metrics.benchmark_metrics import (
    ALL_THREAT_CATEGORIES,
    compute_asr,
    compute_judge_score,
    compute_coverage,
)
from benchmark.ablation.ablation_unalign import AblationUnalign, _DEFAULT_ADAPTER_PATH
from benchmark.ablation.ablation_gpo import SCORE_THRESHOLD, _GPO_REFINE_TEMPLATE

logger = logging.getLogger(__name__)


class CrossThreatTransfer:
    """T3: Evaluate A4 framework across all 12 threat categories.

    For each threat category:
    1. Filter behaviors to only those belonging to the category.
    2. Run M_a → GPO pipeline.
    3. Report per-category ASR and within-category coverage.

    Args:
        target_model: Model name for the target (victim) model.
        optimizer_model: Model name for prompt generation and refinement.
        base_url: vLLM OpenAI-compatible API base URL.
        adapter_path: Path to the LoRA adapter for the unalignment model.
        num_epochs: Number of GPO refinement epochs per behavior.
        judge: LLMJudge instance; created automatically if None.
    """

    def __init__(
        self,
        target_model: str = DEFAULT_MODEL,
        optimizer_model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        adapter_path: str = _DEFAULT_ADAPTER_PATH,
        num_epochs: int = 3,
        judge: LLMJudge | None = None,
    ) -> None:
        self.target_model = target_model
        self.optimizer_model = optimizer_model
        self.base_url = base_url
        self.adapter_path = adapter_path
        self.num_epochs = num_epochs
        self._client = build_openai_client(base_url)
        self._judge = judge or LLMJudge(model=optimizer_model, base_url=base_url)
        self._unalign = AblationUnalign(
            target_model=target_model,
            base_model=optimizer_model,
            base_url=base_url,
            adapter_path=adapter_path,
            judge=self._judge,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attack_behavior(self, behavior: str, threat_category: str) -> dict:
        """Run M_a + GPO for a single behavior."""
        initial_prompt = self._unalign.generate_attack_prompt(behavior, threat_category)
        attack_prompt = initial_prompt
        best_prompt, best_score, best_response = initial_prompt, 1, ""

        for epoch in range(1, self.num_epochs + 1):
            response = call_target_model(
                self._client,
                model=self.target_model,
                prompt=attack_prompt,
                max_tokens=512,
            )
            score = self._judge.score(behavior, attack_prompt, response)

            if score > best_score:
                best_score, best_prompt, best_response = score, attack_prompt, response

            if score >= SCORE_THRESHOLD:
                break

            refine_prompt = _GPO_REFINE_TEMPLATE.format(
                score=score, behavior=behavior, prev_prompt=attack_prompt[:600]
            )
            attack_prompt = call_target_model(
                self._client,
                model=self.optimizer_model,
                prompt=refine_prompt,
                max_tokens=512,
                temperature=1.0,
            )

        return {
            "behavior": behavior,
            "threat_category": threat_category,
            "attack_prompt": best_prompt,
            "response": best_response,
            "score": best_score,
            "success": best_score >= SCORE_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_threat_category(
        self, threat_category: str, behaviors: list[dict]
    ) -> tuple[list[dict], dict]:
        """Run the full pipeline for one threat category.

        Args:
            threat_category: The threat category key.
            behaviors: Only behaviors belonging to this category.

        Returns:
            (results list, metrics dict) pair.
        """
        results: list[dict] = []
        for bdict in behaviors:
            behavior = bdict.get("behavior", bdict.get("goal", ""))
            logger.info(
                "[T3-CrossThreat] threat=%s behavior=%r", threat_category, behavior[:60]
            )
            result = self._attack_behavior(behavior, threat_category)
            results.append(result)

        scores = [r["score"] for r in results]
        successful_threats = [threat_category for r in results if r["success"]]
        metrics = {
            "threat_category": threat_category,
            "n_total": len(results),
            "n_success": sum(1 for r in results if r["success"]),
            "asr": round(compute_asr(scores), 4),
            "js": round(compute_judge_score(scores), 4),
            "coverage": round(
                compute_coverage(ALL_THREAT_CATEGORIES, successful_threats), 4
            ),
        }
        return results, metrics

    def run(self, behaviors: list[dict], output_file: str) -> dict:
        """Run cross-threat-category evaluation across all 12 threat categories.

        Behaviors without a 'threat_category' field are assigned to 'unknown'
        and processed under a single aggregate pass.

        Args:
            behaviors: List of behavior dicts.
            output_file: JSON output path.

        Returns:
            Summary dict keyed by threat category.
        """
        # Group behaviors by threat category
        by_category: dict[str, list[dict]] = {cat: [] for cat in ALL_THREAT_CATEGORIES}
        uncategorised: list[dict] = []
        for bdict in behaviors:
            cat = bdict.get("threat_category", "")
            if cat in by_category:
                by_category[cat].append(bdict)
            else:
                uncategorised.append(bdict)

        if uncategorised:
            logger.warning(
                "%d behaviors have no recognised threat_category — grouped as 'unknown'.",
                len(uncategorised),
            )
            by_category["unknown"] = uncategorised

        summary: dict = {
            "threat_categories": {},
            "total_behaviors": len(behaviors),
            "all_threat_categories": ALL_THREAT_CATEGORIES,
        }

        all_results: list[dict] = []
        for threat_category, cat_behaviors in by_category.items():
            if not cat_behaviors:
                logger.info(
                    "[T3-CrossThreat] No behaviors for %s — skipping.", threat_category
                )
                continue

            logger.info(
                "[T3-CrossThreat] Category=%s  n=%d",
                threat_category,
                len(cat_behaviors),
            )
            try:
                results, metrics = self.run_threat_category(
                    threat_category, cat_behaviors
                )
                summary["threat_categories"][threat_category] = metrics
                all_results.extend(results)
                logger.info(
                    "[T3-CrossThreat] %s ASR=%.4f", threat_category, metrics["asr"]
                )
            except Exception as exc:
                logger.error("Failed for threat_category=%s: %s", threat_category, exc)
                summary["threat_categories"][threat_category] = {"error": str(exc)}

        # Overall coverage: which categories had ≥1 success?
        successful_cats = [
            cat
            for cat, m in summary["threat_categories"].items()
            if isinstance(m, dict) and m.get("n_success", 0) > 0
        ]
        summary["overall_coverage"] = round(
            compute_coverage(ALL_THREAT_CATEGORIES, successful_cats), 4
        )
        summary["overall_asr"] = round(
            compute_asr([r["score"] for r in all_results]) if all_results else 0.0, 4
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        logger.info("Cross-threat results saved to %s", output_file)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="T3: Cross-threat-category transferability."
    )
    parser.add_argument(
        "--behaviors", required=True, help="Path to HarmBench behaviors JSONL."
    )
    parser.add_argument(
        "--output",
        default="results/transferability/cross_threat.json",
        help="Output JSON path.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="GPO refinement epochs.")
    parser.add_argument(
        "--adapter-path",
        default=_DEFAULT_ADAPTER_PATH,
        help="Path to the LoRA adapter for the unalignment model.",
    )
    parser.add_argument(
        "--target-model", default=DEFAULT_MODEL, help="Target model name."
    )
    parser.add_argument(
        "--optimizer-model", default=DEFAULT_MODEL, help="Optimizer model name."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM base URL.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=ALL_THREAT_CATEGORIES,
        help="Subset of threat categories to run (default: all 12).",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _build_parser().parse_args()

    behaviors = load_behaviors(args.behaviors)
    runner = CrossThreatTransfer(
        target_model=args.target_model,
        optimizer_model=args.optimizer_model,
        base_url=args.base_url,
        adapter_path=args.adapter_path,
        num_epochs=args.epochs,
    )
    summary = runner.run(behaviors, output_file=args.output)
    print(f"Done. Cross-threat summary written to {args.output}")
    print(f"Overall ASR: {summary['overall_asr']:.2%}")
    print(f"Overall Coverage: {summary['overall_coverage']:.2%}")
    for cat, m in summary.get("threat_categories", {}).items():
        if "asr" in m:
            print(f"  {cat}: ASR={m['asr']:.2%}, n={m['n_total']}")
