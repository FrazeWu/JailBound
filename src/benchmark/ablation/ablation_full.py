"""A4 Ablation — Full framework (M_a → GPO → QuoTe boundary scoring).

Pipeline:
    1. M_a (unalignment model) generates an initial attack prompt.
    2. GPO optimizer iteratively refines the prompt to maximise ASR.
    3. QuoTe FOL scores (from precomputed quote/outputs/) are used to
       re-rank / filter prompts — those near the decision boundary (high FOL)
       are prioritised.  If QuoTe outputs are unavailable, this step is skipped
       gracefully.

CLI usage:
    python ablation_full.py \\
        --behaviors data/harmbench_behaviors.jsonl \\
        --output results/ablation/ablation_full.jsonl \\
        --epochs 3 \\
        --adapter-path ../attack_model/LLaMA-Factory/saves/qwen3-4b/lora/meta-attack \\
        --quote-outputs ../quote/outputs/
"""

from __future__ import annotations

import argparse
import csv
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
from benchmark.ablation.ablation_unalign import AblationUnalign, _DEFAULT_ADAPTER_PATH
from benchmark.ablation.ablation_gpo import SCORE_THRESHOLD, _GPO_REFINE_TEMPLATE

logger = logging.getLogger(__name__)

# Default QuoTe outputs directory
_DEFAULT_QUOTE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "quote",
    "outputs",
)

_QUOTE_SIGNALS_FILE = "quality_signals.csv"


def _load_quote_fol_scores(quote_outputs_dir: str) -> dict[str, float]:
    """Load precomputed QuoTe FOL scores from CSV.

    The CSV is expected to have columns: behavior (or prompt), fol_score.
    Returns a dict mapping behavior text → FOL score.  Empty dict if file
    does not exist.
    """
    csv_path = os.path.join(quote_outputs_dir, _QUOTE_SIGNALS_FILE)
    if not os.path.isfile(csv_path):
        logger.info(
            "QuoTe signals file not found at %s — skipping FOL filtering.", csv_path
        )
        return {}

    scores: dict[str, float] = {}
    try:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get("behavior") or row.get("prompt", "")
                fol = row.get("fol_score", row.get("fol", ""))
                if key and fol:
                    try:
                        scores[key.strip()] = float(fol)
                    except ValueError:
                        pass
    except Exception as exc:
        logger.warning("Could not load QuoTe signals from %s: %s", csv_path, exc)

    logger.info("Loaded %d QuoTe FOL scores from %s", len(scores), csv_path)
    return scores


class AblationFull:
    """A4 ablation: full framework — M_a + GPO + QuoTe.

    QuoTe FOL scores guide seed selection.  High-FOL behaviors (near the
    decision boundary) are prioritised because small perturbations there
    can cross from refusal to compliance.

    Args:
        target_model: Model name for the target (victim) model.
        optimizer_model: Model name for the GPO optimizer / feedback LLM.
        base_url: vLLM OpenAI-compatible API base URL.
        adapter_path: Path to the LoRA adapter for the unalignment model.
        num_epochs: Number of GPO refinement epochs.
        quote_outputs_dir: Directory containing QuoTe output files.
        judge: LLMJudge instance; created automatically if None.
    """

    def __init__(
        self,
        target_model: str = DEFAULT_MODEL,
        optimizer_model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        adapter_path: str = _DEFAULT_ADAPTER_PATH,
        num_epochs: int = 3,
        quote_outputs_dir: str = _DEFAULT_QUOTE_DIR,
        judge: LLMJudge | None = None,
    ) -> None:
        self.target_model = target_model
        self.optimizer_model = optimizer_model
        self.base_url = base_url
        self.adapter_path = adapter_path
        self.num_epochs = num_epochs
        self.quote_outputs_dir = quote_outputs_dir
        self._client = build_openai_client(base_url)
        self._judge = judge or LLMJudge(model=optimizer_model, base_url=base_url)

        self._unalign = AblationUnalign(
            target_model=target_model,
            base_model=optimizer_model,
            base_url=base_url,
            adapter_path=adapter_path,
            judge=self._judge,
        )
        self._fol_scores = _load_quote_fol_scores(quote_outputs_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fol_score(self, behavior: str) -> float:
        """Return QuoTe FOL score for a behavior; 0.0 if not available."""
        return self._fol_scores.get(behavior.strip(), 0.0)

    def _gpo_refine(
        self, behavior: str, initial_prompt: str, num_epochs: int
    ) -> tuple[str, int, str, int]:
        """Iterative GPO refinement.  Returns (best_prompt, best_score, best_response, epochs)."""
        attack_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 1
        best_response = ""

        for epoch in range(1, num_epochs + 1):
            response = call_target_model(
                self._client,
                model=self.target_model,
                prompt=attack_prompt,
                max_tokens=512,
            )
            score = self._judge.score(behavior, attack_prompt, response)
            logger.info(
                "[A4-Full] epoch=%d behavior=%r score=%d", epoch, behavior[:60], score
            )

            if score > best_score:
                best_score, best_prompt, best_response = score, attack_prompt, response

            if score >= SCORE_THRESHOLD:
                return best_prompt, best_score, best_response, epoch

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

        return best_prompt, best_score, best_response, epoch

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def attack(self, behavior: str, threat_category: str = "unknown") -> dict:
        """Run the full framework for a single behavior.

        Steps:
            A. QuoTe FOL score lookup (guides priority / extra epochs).
            B. M_a generates initial attack prompt.
            C. GPO refines the prompt.

        Args:
            behavior: Target harmful behavior description.
            threat_category: Threat category key.

        Returns:
            Dict with: behavior, threat_category, attack_prompt, initial_prompt,
            response, score, fol_score, epochs, success.
        """
        fol_score = self._get_fol_score(behavior)

        # High FOL → near decision boundary → grant extra epochs
        effective_epochs = self.num_epochs
        if fol_score > 0:
            effective_epochs = (
                self.num_epochs + 1 if fol_score > 0.5 else self.num_epochs
            )
            logger.info(
                "[A4-Full] QuoTe FOL=%.4f for behavior=%r → epochs=%d",
                fol_score,
                behavior[:60],
                effective_epochs,
            )

        # M_a step
        initial_prompt = self._unalign.generate_attack_prompt(behavior, threat_category)

        # GPO step
        best_prompt, best_score, best_response, actual_epochs = self._gpo_refine(
            behavior, initial_prompt, effective_epochs
        )

        return {
            "behavior": behavior,
            "threat_category": threat_category,
            "attack_prompt": best_prompt,
            "initial_prompt": initial_prompt,
            "response": best_response,
            "score": best_score,
            "fol_score": fol_score,
            "epochs": actual_epochs,
            "success": best_score >= SCORE_THRESHOLD,
        }

    def run(
        self,
        behaviors: list[dict],
        output_file: str,
        sort_by_fol: bool = True,
    ) -> list[dict]:
        """Run the full framework on all behaviors.

        Behaviors can be sorted by QuoTe FOL score (descending) so that
        high-boundary cases are processed first.

        Args:
            behaviors: List of behavior dicts.
            output_file: Destination JSONL path.
            sort_by_fol: If True and FOL scores are available, sort behaviors by FOL descending.

        Returns:
            List of result dicts (original order preserved in output).
        """
        if sort_by_fol and self._fol_scores:
            behaviors = sorted(
                behaviors,
                key=lambda b: self._get_fol_score(b.get("behavior", b.get("goal", ""))),
                reverse=True,
            )
            logger.info("Sorted behaviors by QuoTe FOL score (descending).")

        results: list[dict] = []
        for i, bdict in enumerate(behaviors, start=1):
            behavior = bdict.get("behavior", bdict.get("goal", ""))
            threat = bdict.get("threat_category", "unknown")
            logger.info(
                "[A4-Full] %d/%d  behavior=%r", i, len(behaviors), behavior[:60]
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
        description="A4 ablation: full framework (M_a + GPO + QuoTe)."
    )
    parser.add_argument(
        "--behaviors", required=True, help="Path to HarmBench behaviors JSONL."
    )
    parser.add_argument(
        "--output",
        default="results/ablation/ablation_full.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of GPO epochs.")
    parser.add_argument(
        "--adapter-path",
        default=_DEFAULT_ADAPTER_PATH,
        help="Path to the LoRA adapter for the unalignment model.",
    )
    parser.add_argument(
        "--quote-outputs",
        default=_DEFAULT_QUOTE_DIR,
        help="Directory containing QuoTe output files (quality_signals.csv).",
    )
    parser.add_argument(
        "--target-model", default=DEFAULT_MODEL, help="Target model name."
    )
    parser.add_argument(
        "--optimizer-model", default=DEFAULT_MODEL, help="Optimizer model name."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="vLLM base URL.")
    parser.add_argument(
        "--no-fol-sort",
        action="store_true",
        help="Do not sort behaviors by QuoTe FOL score.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _build_parser().parse_args()

    behaviors = load_behaviors(args.behaviors)
    ablation = AblationFull(
        target_model=args.target_model,
        optimizer_model=args.optimizer_model,
        base_url=args.base_url,
        adapter_path=args.adapter_path,
        num_epochs=args.epochs,
        quote_outputs_dir=args.quote_outputs,
    )
    results = ablation.run(
        behaviors, output_file=args.output, sort_by_fol=not args.no_fol_sort
    )
    n_success = sum(1 for r in results if r["success"])
    print(f"Done. {n_success}/{len(results)} successes. Output: {args.output}")
