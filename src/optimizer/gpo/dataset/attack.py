"""Attack dataset for GPO-based jailbreak prompt optimization.

Wraps attack_sft_v2_sample100_output.json so it fits the GPO dataset interface.

Design:
  - Each "task" = one attack type (e.g., persuasion_deception).
  - Each data item's "question" = the full attack prompt text.
  - The "instruction" optimized by GPO = a refinement suffix appended after
    the attack prompt to improve jailbreak effectiveness.
  - "true_answer" = "" (empty); scoring is done externally by the judge.
  - prediction_treat_as_rouge=True so eval_utils uses calc_rouge which we
    monkeypatch to call the jailbreak judge instead.
"""

import json
import os
from collections import defaultdict

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

# Canonical attack type keys (normalized from dataset)
ATTACK_TYPE_NORMALIZE = {
    "Obfuscation & Encryption": "obfuscation_encryption",
    "Scenario Nesting": "scenario_nesting",
    "Compositional / Hybrid Attacks": "compositional_hybrid",
    "Persuasion & Deception": "persuasion_deception",
    "Code Injection": "code_injection",
    "Input Fragmentation": "input_fragmentation",
    "Prefix Injection": "prefix_injection",
}

# Default path — can be overridden via env var ATTACK_DATASET_PATH
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(GPO_ROOT_PATH),
    "attack_model",
    "data",
    "attack_sft_v2_sample100_output.json",
)

# Sample limit — can be set via env var ATTACK_SAMPLE_LIMIT (0 = no limit)
SAMPLE_LIMIT = int(os.getenv("ATTACK_SAMPLE_LIMIT", "0"))


def _load_and_group(dataset_path: str) -> dict[str, list[dict]]:
    """Load JSON and group records by normalized attack type."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Apply global sample limit if set
    sample_limit = SAMPLE_LIMIT
    if sample_limit > 0:
        raw = raw[:sample_limit]
        print(f"[Attack Dataset] Applied sample limit: {sample_limit}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in raw:
        instr = record.get("instruction", "")
        attack_type_raw = "unknown"
        if "Attack Type:" in instr:
            attack_type_raw = instr.split("Attack Type:")[1].split(",")[0].strip()
        attack_type = ATTACK_TYPE_NORMALIZE.get(
            attack_type_raw,
            attack_type_raw.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("&", "")
            .replace("__", "_"),
        )
        # Flatten record for easy access
        groups[attack_type].append(
            {
                "id": record.get("id"),
                "input": record.get("prompt", ""),  # the attack prompt text
                "target": "",  # no fixed ground truth
                "instruction_meta": instr,  # original instruction field
                "source": record.get("source", ""),
            }
        )
    return dict(groups)


class AttackDataset:
    """GPO-compatible dataset wrapping attack_sft_v2_sample100_output.json."""

    def __init__(self, dataset_path: str | None = None) -> None:
        self.dataset_path = dataset_path or os.getenv(
            "ATTACK_DATASET_PATH", DEFAULT_DATASET_PATH
        )
        self._groups: dict[str, list[dict]] | None = None

    def _ensure_loaded(self) -> dict[str, list[dict]]:
        if self._groups is None:
            self._groups = _load_and_group(self.dataset_path)
        return self._groups

    # ------------------------------------------------------------------
    # GPO interface
    # ------------------------------------------------------------------

    def get_ratio(self) -> tuple[float, float, float]:
        """Train/eval/test split ratios (all 1.0 → use all data for each)."""
        train_ratio = 0.6
        eval_ratio = 0.2
        test_ratio = 0.2
        print(
            f"[Attack Dataset] train_ratio={train_ratio}, "
            f"eval_ratio={eval_ratio}, test_ratio={test_ratio}"
        )
        return train_ratio, eval_ratio, test_ratio

    def read_data(self, task: list[str] | str) -> list[dict]:
        """Return list of task-dicts compatible with optimize.py."""
        groups = self._ensure_loaded()
        all_tasks = sorted(groups.keys())

        if isinstance(task, list) and task and task[0] == "all":
            selected_tasks = all_tasks
        elif isinstance(task, str):
            assert task in all_tasks or task == "all", (
                f"Unknown attack task '{task}'. Available: {all_tasks}"
            )
            selected_tasks = all_tasks if task == "all" else [task]
        elif isinstance(task, list):
            for t in task:
                assert t in all_tasks, (
                    f"Unknown attack task '{t}'. Available: {all_tasks}"
                )
            selected_tasks = task
        else:
            raise ValueError(f"Invalid task argument: {task!r}")

        datas = []
        for t in selected_tasks:
            items = groups[t]
            n = len(items)

            # Train / eval / test / format / few_shot all share the same data.
            # We use the full list for each split; optimize.py will sample via ratios.
            datas.append(
                {
                    "task": t,
                    "train_data": items,
                    "eval_data": items,
                    "test_data": items,
                    "format_data": items,
                    "few_shot_data": items,
                    "train_num_examples": n,
                    "eval_num_examples": n,
                    "test_num_examples": n,
                    "format_num_examples": n,
                    "few_shot_num_examples": n,
                }
            )
            print(f"[Attack Dataset] task={t}, n={n} examples")

        return datas

    def get_single_question(self, data: list[dict], idx: int) -> str:
        """Return the attack prompt text as the 'question'."""
        return data[idx]["input"]

    def get_single_answer(self, data: list[dict], idx: int) -> str:
        """Return empty string; jailbreak success is judged externally."""
        return data[idx]["target"]  # ""

    def get_single_solution(self, data: list[dict], idx: int) -> None:
        return None

    def get_q(self, data: dict) -> str:
        return data["input"]

    def get_a(self, data: dict) -> str:
        return data["target"]

    def get_task_setting(self, task: str) -> dict:
        """Task settings for attack prompts.

        We use prediction_treat_as_rouge=True so that eval_utils.py calls
        metrics.calc_rouge(response, true_answer). Since true_answer="" the
        ROUGE score will always be 0 — but we monkeypatch calc_rouge in
        the run script to call our jailbreak judge instead.
        """
        return {
            "is_multiple_choice": False,
            "prediction_treat_as_number": False,
            "prediction_treat_as_bool": False,
            "prediction_treat_as_rouge": True,
            "extract_final_answer_by_prompting_again": False,
        }
