"""
QuoTe Configuration Module

Hyperparameters and CLI argument parsing for the compliance boundary detection pipeline.
All defaults follow the QuoTe paper (Chen et al., TOSEM 2023) adapted for LLM safety evaluation.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import ClassVar


# =============================================================================
# Default constants
# =============================================================================

DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET_PATH = "data/benchmark/harmbench_behaviors.jsonl"
DEFAULT_OUTPUT_DIR = "quote/outputs"


# =============================================================================
# QuoTe hyperparameter dataclass
# =============================================================================


@dataclass
class QuoTeConfig:
    """
    Configuration for the QuoTe compliance boundary detection pipeline.

    Attributes:
        model_path: Path or HuggingFace ID of the target model (white-box, local).
        dataset_path: Path to HarmBench behaviors CSV file.
        output_dir: Directory for JSON/CSV output metrics.
        epsilon: L2 perturbation ball radius (ε). Controls max perturbation magnitude.
        eta: Gradient step size (η). Learning rate for the iterative search.
        steps: Number of perturbation steps per seed sample.
        k: Number of segments for KM-ST (k-means stratified) seed selection.
        alpha: Top-α fraction for BE-ST high-FOL seed selection (0–1).
        beta: Bottom-β fraction for BE-ST low-FOL seed selection (0–1).
        grad_clip: Global gradient L2 clipping threshold.
        batch_size: Number of behaviors to process simultaneously.
        max_new_tokens: Max tokens when sampling model completions for verification.
        threat_filter: If set, only process behaviors matching this threat category ID.
        functional_filter: HarmBench functional category filter ("standard", "contextual", or "all").
        seed: Random seed for reproducibility.
        device: Torch device string ("cuda", "cuda:0", "cpu", etc.).
        torch_dtype: Model loading dtype ("bfloat16", "float16", "float32").
    """

    # Model and data
    model_path: str = DEFAULT_MODEL_PATH
    dataset_path: str = DEFAULT_DATASET_PATH
    output_dir: str = DEFAULT_OUTPUT_DIR

    # Core QuoTe hyperparameters
    epsilon: float = 0.1  # L2 perturbation ball radius
    eta: float = 0.01  # gradient step size
    steps: int = 100  # perturbation iterations per seed
    k: int = 5  # KM-ST segment count
    alpha: float = 0.10  # BE-ST high-FOL fraction
    beta: float = 0.10  # BE-ST low-FOL fraction
    grad_clip: float = 1.0  # global gradient clipping

    # Runtime
    batch_size: int = 1
    max_new_tokens: int = 128
    threat_filter: str | None = None
    functional_filter: str = "standard"
    seed: int = 42
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_behaviors: int | None = None  # if set, cap the number of behaviors processed

    # -------------------------------------------------------------------------
    # Soft-prefix optimization parameters  (v2 pipeline)
    # -------------------------------------------------------------------------
    prefix_length: int = 20         # number of learnable continuous prefix tokens (z)
    lr: float = 1e-3                # Adam learning rate for [z, U]
    max_opt_steps: int = 200        # total optimisation steps per branch
    gamma_z: float = 0.01           # prefix z L2² regularisation weight
    gamma_u: float = 0.01           # editable seed block U drift penalty weight
    lambda_fol: float = 0.1         # FOL weight in branch objectives
    alpha_score: float = 1.0        # risk weight in candidate selection score
    beam_size: int = 4              # beam / candidate pool size per branch
    early_stop_patience: int = 30   # steps without improvement before stopping

    # Local model paths
    judge_model_path: str = ""      # empty = use default benchmark/judge_model
    wildguard_model_path: str = os.environ.get("WILDGUARD_MODEL_PATH", "models/wildguard")
    target_models: list[str] = field(default_factory=lambda: [
        "models/surrogate-small",
        "models/surrogate-large",
    ])

    # Proxy calibration: decode + WildGuard judge every N steps
    judge_interval: int = 25        # steps between WildGuard proxy evaluations

    # Ablation mode: full | no_fol | high_value_only | boundary_only | init_only
    ablation_mode: str = "full"

    # Candidate selection weights
    select_top_k: int = 10
    diversity_weight: float = 0.1
    hv_boundary_ratio: float = 0.5  # mix ratio: high-value vs boundary pool

    _VALID_ABLATIONS: ClassVar[tuple[str, ...]] = (
        "full", "no_fol", "high_value_only", "boundary_only", "init_only",
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        assert self.epsilon > 0, f"epsilon must be positive, got {self.epsilon}"
        assert self.eta > 0, f"eta must be positive, got {self.eta}"
        assert self.steps > 0, f"steps must be positive, got {self.steps}"
        assert self.k >= 2, f"k must be >= 2 for stratification, got {self.k}"
        assert 0.0 < self.alpha <= 1.0, f"alpha must be in (0, 1], got {self.alpha}"
        assert 0.0 < self.beta <= 1.0, f"beta must be in (0, 1], got {self.beta}"
        assert self.alpha + self.beta <= 1.0, (
            f"alpha + beta must not exceed 1.0, got {self.alpha + self.beta}"
        )
        assert self.grad_clip > 0, f"grad_clip must be positive, got {self.grad_clip}"
        assert self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}"
        assert self.prefix_length >= 1, f"prefix_length must be >= 1, got {self.prefix_length}"
        assert self.lr > 0, f"lr must be positive, got {self.lr}"
        assert self.max_opt_steps >= 1, f"max_opt_steps must be >= 1, got {self.max_opt_steps}"
        assert 0.0 <= self.gamma_z, f"gamma_z must be >= 0, got {self.gamma_z}"
        assert 0.0 <= self.gamma_u, f"gamma_u must be >= 0, got {self.gamma_u}"
        assert 0.0 <= self.lambda_fol, f"lambda_fol must be >= 0, got {self.lambda_fol}"
        assert self.judge_interval >= 1, f"judge_interval must be >= 1, got {self.judge_interval}"
        assert self.ablation_mode in self._VALID_ABLATIONS, (
            f"ablation_mode must be one of {self._VALID_ABLATIONS}, got '{self.ablation_mode}'"
        )


# =============================================================================
# CLI argument parser
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for run_quote.py."""
    parser = argparse.ArgumentParser(
        prog="run_quote",
        description=(
            "QuoTe: Compliance Boundary Detection via Embedding-Space Perturbation.\n"
            "Locates safety/refusal decision boundaries in a local white-box LLM "
            "by computing zero-order and first-order quality signals on HarmBench behaviors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Model & data
    # -------------------------------------------------------------------------
    io_group = parser.add_argument_group("Model & Data")
    io_group.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="HuggingFace model ID or local path of the target model.",
    )
    io_group.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the HarmBench behaviors CSV file.",
    )
    io_group.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON/CSV result files will be written.",
    )

    # -------------------------------------------------------------------------
    # QuoTe hyperparameters
    # -------------------------------------------------------------------------
    hparam_group = parser.add_argument_group("QuoTe Hyperparameters")
    hparam_group.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="L2 perturbation ball radius ε. Controls max embedding perturbation magnitude.",
    )
    hparam_group.add_argument(
        "--eta",
        type=float,
        default=0.01,
        help="Gradient step size η for iterative perturbation search.",
    )
    hparam_group.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of perturbation steps per seed sample.",
    )
    hparam_group.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of segments for KM-ST (k-means stratified) seed selection.",
    )
    hparam_group.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Top-α fraction (0–1) of high-FOL samples selected by BE-ST.",
    )
    hparam_group.add_argument(
        "--beta",
        type=float,
        default=0.10,
        help="Bottom-β fraction (0–1) of low-FOL samples selected by BE-ST.",
    )
    hparam_group.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Global L2 gradient clipping threshold.",
    )

    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------
    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of behaviors processed simultaneously.",
    )
    runtime_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens when sampling model completions for verification.",
    )
    runtime_group.add_argument(
        "--threat_filter",
        type=str,
        default=None,
        help=(
            "If set, only process HarmBench behaviors whose SemanticCategory maps to "
            "this threat category ID (e.g. 'cybersecurity_misuse')."
        ),
    )
    runtime_group.add_argument(
        "--functional_filter",
        type=str,
        default="standard",
        choices=["standard", "contextual", "all"],
        help="HarmBench FunctionalCategory filter.",
    )
    runtime_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    runtime_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    runtime_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model loading dtype.",
    )
    runtime_group.add_argument(
        "--max_behaviors",
        type=int,
        default=None,
        help="Cap the number of behaviors to process (useful for smoke tests).",
    )

    return parser


def parse_args(argv: list[str] | None = None) -> QuoTeConfig:
    """
    Parse CLI arguments and return a validated QuoTeConfig.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Populated and validated QuoTeConfig instance.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return QuoTeConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        eta=args.eta,
        steps=args.steps,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        threat_filter=args.threat_filter,
        functional_filter=args.functional_filter,
        seed=args.seed,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_behaviors=args.max_behaviors,
    )
