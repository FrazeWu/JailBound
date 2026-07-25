#!/usr/bin/env python3
"""Annotate downloaded benchmark datasets into language-split Alpaca files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from corpora.alpaca_annotation.pipeline import PipelineConfig, run_pipeline
from corpora.alpaca_annotation.vllm import VLLMAnnotator, VLLMServer


DEFAULT_INPUT_DIR = Path("01benchmark-datasets/downloaded_datasets")
DEFAULT_OUTPUT_DIR = Path("data/alpaca_annotation")
DEFAULT_MODEL_PATH = Path(os.environ.get("ANNOTATION_MODEL_PATH", "models/qwen"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vllm-url", default=None, help="Existing vLLM base URL, for example http://localhost:8000")
    parser.add_argument("--auto-start-vllm", action="store_true")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=900)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--language-quotas", default=None, help="Comma-separated quotas, for example zh=15000,en=15000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--datasets", default=None, help="Comma-separated top-level dataset names")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = {item.strip() for item in args.datasets.split(",") if item.strip()} if args.datasets else None
    language_quotas = _parse_language_quotas(args.language_quotas)
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        language_quotas=language_quotas,
        datasets=datasets,
        scan_only=args.scan_only,
        force=args.force,
        workers=args.workers,
    )

    server: VLLMServer | None = None
    annotator = None
    try:
        if not args.scan_only:
            if args.auto_start_vllm:
                server = VLLMServer(
                    model_path=args.model_path,
                    port=args.port,
                    gpu=args.gpu,
                    tensor_parallel_size=args.tensor_parallel_size,
                    log_path=args.output_dir / f"vllm_{args.model_path.name}.log",
                )
                server.start(timeout=args.vllm_startup_timeout)
                base_url = server.base_url
                model_name = server.model_name
            else:
                base_url = args.vllm_url or f"http://localhost:{args.port}"
                model_name = args.model_name or args.model_path.name
            annotator = VLLMAnnotator(base_url=base_url, model=model_name)

        manifest = run_pipeline(config, annotator)
        summary = manifest["summary"]
        print(
            "Done: "
            f"{summary['converted_records']} converted, "
            f"{summary['duplicates_removed']} duplicates removed, "
            f"{summary['annotation_errors']} annotation errors."
        )
    finally:
        if server is not None:
            server.stop()


def _parse_language_quotas(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    quotas: dict[str, int] = {}
    for item in value.split(","):
        language, sep, raw_count = item.strip().partition("=")
        if not sep:
            raise ValueError(f"invalid language quota {item!r}; expected language=count")
        if language not in {"en", "zh", "multilingual", "unknown"}:
            raise ValueError(f"unsupported language quota {language!r}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"language quota for {language!r} must be non-negative")
        quotas[language] = count
    return quotas


if __name__ == "__main__":
    main()
