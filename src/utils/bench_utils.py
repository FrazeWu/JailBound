"""Shared utilities for the benchmark suite.

Provides:
    - load_behaviors()        : JSONL loader for HarmBench behaviors
    - call_target_model()     : OpenAI-compatible API call to vLLM target
    - save_jsonl()            : Append results to a JSONL file
    - build_openai_client()   : Construct an OpenAI client from env / defaults
"""

from __future__ import annotations

import json
import logging
import os
import sys

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("BENCHMARK_API_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = "qwen-72b"


def build_openai_client(base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Create an OpenAI client pointed at the vLLM endpoint.

    The API key is read from $OPENAI_API_KEY (defaults to 'EMPTY', which
    vLLM accepts without authentication).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(api_key=api_key, base_url=base_url)


def load_behaviors(behaviors_file: str) -> list[dict]:
    """Load a JSONL behaviors file.

    Each line must be a JSON object.  Common keys:
        behavior      (str) — the target harmful behavior description
        threat_category (str) — one of the 12 threat category keys (optional)
        attack_type   (str) — one of the 8 attack type keys (optional)

    Args:
        behaviors_file: Path to the JSONL file.

    Returns:
        List of behavior dicts.

    Raises:
        FileNotFoundError: If *behaviors_file* does not exist.
    """
    if not os.path.isfile(behaviors_file):
        raise FileNotFoundError(f"Behaviors file not found: {behaviors_file}")

    records: list[dict] = []
    with open(behaviors_file, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed line %d in %s: %s", lineno, behaviors_file, exc
                )
    return records


def call_target_model(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 512,
    disable_thinking: bool = True,
) -> str:
    """Send a prompt to the target model and return its text response.

    Args:
        client: OpenAI client instance.
        model: Model name (as served by vLLM).
        prompt: User-facing prompt text.
        system_prompt: Optional system prompt prepended as a system message.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        disable_thinking: If True, passes extra_body to disable Qwen3 thinking.

    Returns:
        Response text string; empty string on failure.
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    extra_body: dict = {}
    if disable_thinking:
        extra_body["thinking"] = {"type": "disabled"}

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Target model call failed: %s", exc)
        return ""


def save_jsonl(records: list[dict], output_file: str) -> None:
    """Write (or overwrite) a JSONL output file.

    Args:
        records: List of dicts to serialise, one per line.
        output_file: Destination path (parent directories created as needed).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), output_file)
