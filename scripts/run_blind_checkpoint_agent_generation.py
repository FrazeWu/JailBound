"""Generate blind agent outputs from checkpoint prompts and their token embeddings.

The checkpoint prompts and generated responses are treated as opaque strings:
this runner neither prints nor evaluates their contents. Each source checkpoint
file produces separate prompt-input and embedding-input output ledgers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/wh/models/qwen/Qwen2___5-7B-Instruct")
DEFAULT_UTF8_CHECKPOINTS = ROOT / "outputs/blind_alpaca_embedding_roundtrip/run_20260728_visible_frozen_utf8/checkpoints.jsonl"
DEFAULT_ASCII_CHECKPOINTS = ROOT / "outputs/blind_alpaca_embedding_roundtrip/run_20260728_ascii_english/checkpoints.jsonl"


def checkpoint_source_paths(
    selected_sources: list[str] | None, utf8_checkpoints: Path, ascii_checkpoints: Path
) -> dict[str, Path]:
    """Return only the requested checkpoint sources in a stable order."""
    available = {
        "utf8_multilingual": utf8_checkpoints,
        "ascii_english": ascii_checkpoints,
    }
    names = selected_sources or list(available)
    if set(names) - set(available):
        raise ValueError("requested checkpoint source is unknown")
    return {name: available[name] for name in names}


def load_checkpoint_records(path: Path) -> list[dict[str, Any]]:
    """Load structurally valid checkpoint prompts without inspecting their text."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError(f"checkpoint file is empty: {path}")
    if any(set(record) != {"step", "prompt"} for record in records):
        raise ValueError(f"checkpoint file has an unexpected schema: {path}")
    if any(not isinstance(record["step"], int) or not isinstance(record["prompt"], str) for record in records):
        raise ValueError(f"checkpoint file contains invalid value types: {path}")
    return records


def generate_checkpoint_outputs(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate once from token IDs and once from the corresponding embeddings."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    text_records: list[dict[str, Any]] = []
    embedding_records: list[dict[str, Any]] = []

    for record in records:
        prompt = record["prompt"]
        step = record["step"]
        assert isinstance(prompt, str)
        assert isinstance(step, int)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)

        with torch.inference_mode():
            text_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            inputs_embeds = embedding_layer(input_ids).detach()
            embedding_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        text_response = tokenizer.decode(text_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
        embedding_response = tokenizer.decode(embedding_ids[0], skip_special_tokens=True)
        text_records.append({"step": step, "output": text_response})
        embedding_records.append({"step": step, "output": embedding_response})

    return text_records, embedding_records


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--utf8-checkpoints", type=Path, default=DEFAULT_UTF8_CHECKPOINTS)
    parser.add_argument("--ascii-checkpoints", type=Path, default=DEFAULT_ASCII_CHECKPOINTS)
    parser.add_argument("--source", action="append", choices=("utf8_multilingual", "ascii_english"))
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/blind_alpaca_embedding_roundtrip/agent_generations"
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if not args.agent_model.is_dir():
        raise FileNotFoundError(args.agent_model)
    source_paths = checkpoint_source_paths(args.source, args.utf8_checkpoints, args.ascii_checkpoints)
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    sources = {source_name: load_checkpoint_records(path) for source_name, path in source_paths.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.agent_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.agent_model, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    for source_name, records in sources.items():
        text_records, embedding_records = generate_checkpoint_outputs(
            model, tokenizer, records, max_new_tokens=args.max_new_tokens
        )
        destination = args.output_dir / source_name
        write_records(destination / "prompt_input_outputs.jsonl", text_records)
        write_records(destination / "embedding_input_outputs.jsonl", embedding_records)

    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
