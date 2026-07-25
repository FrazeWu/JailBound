"""A3 Ablation — GPO + Unalignment model (no QuoTe).

Pipeline:
    1. M_a (unalignment model) generates an initial attack prompt.
    2. GPO optimizer iteratively refines the prompt to maximise ASR.
    3. Best prompt is evaluated against the target model.

This combines A2 (unalignment) and A1 (GPO) without the QuoTe boundary
scoring used in the full framework (A4).

CLI usage:
    python ablation_gpo_unalign.py \\
        --behaviors data/harmbench_behaviors.jsonl \\
        --output results/ablation/ablation_gpo_unalign.jsonl \\
        --epochs 3 \\
        --adapter-path ../attack_model/LLaMA-Factory/saves/qwen3-4b/lora/meta-attack
"""

from __future__ import annotations

import argparse
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

# Re-use components from sibling ablation modules
from benchmark.ablation.ablation_unalign import AblationUnalign, _DEFAULT_ADAPTER_PATH
from benchmark.ablation.ablation_gpo import AblationGPO, _GPO_REFINE_TEMPLATE, SCORE_THRESHOLD

logger = logging.getLogger(__name__)


class AblationGPOUnalign:
    """A3 ablation: unalignment model M_a → GPO optimisation.

    M_a produces an initial attack prompt from the behavior description.
    GPO then refines that prompt over multiple epochs to maximise the judge
    score on the target model.

    Args:
        target_model: Model name for the target (victim) model.
        optimizer_model: Model name for the GPO optimizer / feedback LLM.
        base_url: vLLM OpenAI-compatible API base URL.
        adapter_path: Path to the LoRA adapter for the unalignment model.
        num_epochs: Number of GPO refinement epochs.
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

        # Sub-components
        self._unalign = AblationUnalign(
            target_model=target_model,
            base_model=optimizer_model,
            base_url=base_url,
            adapter_path=adapter_path,
            judge=self._judge,
        )
        self._gpo = AblationGPO(
            target_model=target_model,
            optimizer_model=optimizer_model,
            base_url=base_url,
            num_epochs=num_epochs,
            judge=self._judge,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attack(self, behavior: str, threat_category: str = "unknown") -> dict:
        """M_a generate → GPO refine for a single behavior.

        Args:
            behavior: Target harmful behavior description.
            threat_category: Threat category key.

        Returns:
            Dict with: behavior, threat_category, attack_prompt, response, score, success, epochs.
        """
        # Step 1: M_a generates initial attack prompt
        initial_prompt = self._unalign.generate_attack_prompt(behavior, threat_category)
        logger.info(
            "[A3] Initial prompt from M_a (len=%d) behavior=%r",
            len(initial_prompt),
            behavior[:60],
        )

        # Step 2: GPO refines it
        best_prompt = initial_prompt
        best_score = 1
        best_response = ""

        attack_prompt = initial_prompt
        for epoch in range(1, self.num_epochs + 1):
            response = call_target_model(
                self._client,
                model=self.target_model,
                prompt=attack_prompt,
                max_tokens=512,
            )
            score = self._judge.score(behavior, attack_prompt, response)
            logger.info(
                "[A3-GPO+Unalign] epoch=%d behavior=%r score=%d",
                epoch,
                behavior[:60],
                score,
            )

            if score > best_score:
                best_score, best_prompt, best_response = score, attack_prompt, response

            if score >= SCORE_THRESHOLD:
                break

            # GPO feedback-based refinement
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
            "initial_prompt": initial_prompt,
            "response": best_response,
            "score": best_score,
            "epochs": epoch,
            "success": best_score >= SCORE_THRESHOLD,
        }

    def run(self, behaviors: list[dict], output_file: str) -> list[dict]:
        """Run GPO+Unalign ablation on all behaviors.

        Args:
            behaviors: List of behavior dicts.
            output_file: Destination JSONL path.

        Returns:
            List of result dicts.
        """
        results: list[dict] = []
        for i, bdict in enumerate(behaviors, start=1):
            behavior = bdict.get("behavior", bdict.get("goal", ""))
            threat = bdict.get("threat_category", "unknown")
            logger.info(
                "[A3-GPO+Unalign] %d/%d  behavior=%r", i, len(behaviors), behavior[:60]
            )

            result = self.attack(behavior, threat_category=threat)
            results.append(result)

        save_jsonl(results, output_file)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A3 ablation: GPO + Unalignment model."
    )
    parser.add_argument(
        "--behaviors", required=True, help="Path to HarmBench behaviors JSONL."
    )
    parser.add_argument(
        "--output",
        default="results/ablation/ablation_gpo_unalign.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of GPO epochs.")
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
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _build_parser().parse_args()

    behaviors = load_behaviors(args.behaviors)
    ablation = AblationGPOUnalign(
        target_model=args.target_model,
        optimizer_model=args.optimizer_model,
        base_url=args.base_url,
        adapter_path=args.adapter_path,
        num_epochs=args.epochs,
    )
    results = ablation.run(behaviors, output_file=args.output)
    n_success = sum(1 for r in results if r["success"])
    print(f"Done. {n_success}/{len(results)} successes. Output: {args.output}")
