from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.corpora.annotate_downloaded_to_alpaca import build_parser


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.input_dir == Path("01benchmark-datasets/downloaded_datasets")
    assert args.output_dir == Path("data/alpaca_annotation")
    assert args.model_path == Path("models/qwen")
    assert args.port == 8000
    assert args.gpu == "0"
    assert args.vllm_startup_timeout == 900
    assert args.workers == 1


def test_parser_dataset_list() -> None:
    args = build_parser().parse_args(["--datasets", "AdvBench,StrongREJECT"])

    assert args.datasets == "AdvBench,StrongREJECT"


def test_parser_language_quotas() -> None:
    args = build_parser().parse_args(["--language-quotas", "zh=15000,en=15000"])

    assert args.language_quotas == "zh=15000,en=15000"


def test_parser_workers() -> None:
    args = build_parser().parse_args(["--workers", "6"])

    assert args.workers == 6
