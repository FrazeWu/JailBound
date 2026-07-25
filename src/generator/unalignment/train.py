"""
train.py — Unalignment model SFT training entry point.

Wraps LLaMA-Factory CLI to train the M_a meta-attacker model via LoRA SFT.

Usage:
    # Standard training (uses default config)
    python train.py

    # Custom config
    python train.py --config configs/lora_sft.yaml

    # Quick smoke-test with tiny batch (verifies setup without full training)
    python train.py --smoke-test

    # Resume from checkpoint
    python train.py --resume-from-checkpoint saves/qwen3-4b/lora/meta-attack
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/generator/unalignment/train.py → project root
_LLAMA_FACTORY_ROOT = _PROJECT_ROOT / "third_party" / "LLaMA-Factory"

logger = logging.getLogger(__name__)


def _check_llamafactory() -> bool:
    """Check if LLaMA-Factory CLI is available."""
    result = subprocess.run(
        ["llamafactory-cli", "--help"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _check_dataset(config_path: Path) -> None:
    """Validate that the training dataset file exists."""
    import yaml  # type: ignore

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg.get("dataset", "attack")
    dataset_info_path = _LLAMA_FACTORY_ROOT / "data" / "dataset_info.json"
    mydata_info_path = _LLAMA_FACTORY_ROOT / "mydata" / "dataset_info.json"

    for info_path in [dataset_info_path, mydata_info_path]:
        if info_path.exists():
            import json

            with open(info_path) as f:
                info = json.load(f)
            if dataset_name in info:
                file_name = info[dataset_name].get("file_name", "")
                data_dir = info_path.parent
                data_file = data_dir / file_name
                if not data_file.exists():
                    raise FileNotFoundError(
                        f"Dataset file not found: {data_file}\n"
                        f"Run build_sft_data.py first to generate the training data."
                    )
                logger.info("Dataset '%s' found at %s", dataset_name, data_file)
                return

    logger.warning(
        "Could not validate dataset '%s' — dataset_info.json not found.",
        dataset_name,
    )


def train(
    config_path: Path,
    resume_from_checkpoint: str | None = None,
    smoke_test: bool = False,
    extra_args: list[str] | None = None,
) -> int:
    """
    Run LLaMA-Factory SFT training.

    Args:
        config_path: Path to the YAML training config.
        resume_from_checkpoint: Optional path to resume from.
        smoke_test: If True, override epochs=0.01 and max_steps=5 for quick test.
        extra_args: Additional CLI args passed to llamafactory-cli.

    Returns:
        Exit code of the training process.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if not _check_llamafactory():
        logger.error(
            "llamafactory-cli not found. Install with:\n"
            '  cd %s && pip install -e ".[torch,metrics]"',
            _LLAMA_FACTORY_ROOT,
        )
        return 1

    _check_dataset(config_path)

    cmd = ["llamafactory-cli", "train", str(config_path)]

    if resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", resume_from_checkpoint]

    if smoke_test:
        logger.info("Smoke-test mode: overriding to max_steps=5")
        cmd += [
            "--max_steps",
            "5",
            "--num_train_epochs",
            "1",
            "--logging_steps",
            "1",
            "--save_steps",
            "5",
        ]

    if extra_args:
        cmd += extra_args

    logger.info("Running: %s", " ".join(cmd))

    # Run in LLaMA-Factory directory so relative paths work
    result = subprocess.run(cmd, cwd=str(_LLAMA_FACTORY_ROOT))
    return result.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train M_a unalignment model via LLaMA-Factory SFT."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_HERE / "configs" / "lora_sft.yaml",
        help="Path to training YAML config (default: configs/lora_sft.yaml)",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume training from this checkpoint directory",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run with max_steps=5 to verify setup (no full training)",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to llamafactory-cli train",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    exit_code = train(
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint,
        smoke_test=args.smoke_test,
        extra_args=args.extra or None,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
