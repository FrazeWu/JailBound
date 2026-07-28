"""Run one bounded, harmless prompt-to-embedding-to-prompt demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/alpaca_annotation_30k_zh_en/alpaca_en.json"
MODEL = Path("/home/wh/models/qwen/Qwen2___5-7B-Instruct")
SAFE_SUFFIX = " Please provide a concise and clear answer."
SAFE_TERMS = ("summarize", "list", "describe", "explain", "translate", "poem", "story", "recipe", "history")
EXCLUDED_TERMS = (
    "weapon", "explosive", "malware", "hack", "illegal", "steal", "harm", "kill",
    "suicide", "drug", "crime", "violent", "sexual", "attack",
)


def _safe_candidate(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    instruction = row.get("instruction")
    context = row.get("input")
    output = row.get("output")
    if not all(isinstance(value, str) for value in (instruction, context, output)):
        return False
    combined = f"{instruction} {context} {output}".lower()
    return (
        any(term in instruction.lower() for term in SAFE_TERMS)
        and not any(term in combined for term in EXCLUDED_TERMS)
        and 40 <= len(output) <= 600
    )


def _select_sample(rows: list[object], seed: int) -> dict[str, str]:
    candidates = [row for row in rows if _safe_candidate(row)]
    if not candidates:
        raise ValueError("no harmless Alpaca candidate passed the fixed filter")
    selected = random.Random(seed).choice(candidates)
    assert isinstance(selected, dict)
    return {field: str(selected[field]) for field in ("instruction", "input", "output")}


def _prompt(row: dict[str, str]) -> str:
    context = f"\nContext: {row['input']}" if row["input"].strip() else ""
    return f"Instruction: {row['instruction']}{context}\nAnswer:"


def _nearest_allowed_tokens(embeddings: torch.Tensor, allowed_embeddings: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    allowed = torch.nn.functional.normalize(allowed_embeddings.float(), dim=-1)
    return (normalized @ allowed.T).argmax(dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/safe_alpaca_embedding_demo")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--safe-surrogate", action="store_true", help="use a benign public-safety proxy without retaining unsafe source text")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.safe_surrogate:
        sample = {
            "instruction": "Give concise, practical tips for making a cake in New York City.",
            "input": "",
            "output": "Choose a simple recipe, gather common ingredients, preheat the oven, and check the cake with a toothpick before serving.",
        }
        source = "user_supplied_benign_cake_example"
    else:
        rows = json.loads(DATASET.read_text(encoding="utf-8"))
        sample = _select_sample(rows, args.seed)
        source = "alpaca_en_harmless_allowlist"
    initial_prompt = _prompt(sample)
    initial_hash = hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest()
    (args.output_dir / "01_selected_sample.json").write_text(
        json.dumps({"seed": args.seed, "source": source, "sample": sample}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager"
    ).eval()
    device = model.get_input_embeddings().weight.device
    initial_ids = tokenizer(initial_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    suffix_ids = tokenizer(SAFE_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    embedding_layer = model.get_input_embeddings()
    initial_embeds = embedding_layer(initial_ids).detach()
    suffix_target = embedding_layer(suffix_ids).detach()
    (args.output_dir / "02_embedding.json").write_text(
        json.dumps({
            "initial_prompt_sha256": initial_hash,
            "initial_token_count": int(initial_ids.shape[1]),
            "embedding_shape": list(initial_embeds.shape),
            "embedding_dtype": str(initial_embeds.dtype),
            "safe_suffix_token_count": int(suffix_ids.shape[1]),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    # Optimize only a safe suffix embedding towards the selected concise-answer template.
    state = (suffix_target + 0.02 * torch.randn_like(suffix_target)).detach().float().requires_grad_(True)
    optimizer = torch.optim.SGD([state], lr=0.25)
    losses: list[float] = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = (state - suffix_target.float()).square().mean() + 0.01 * (state - suffix_target.float()).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    (args.output_dir / "03_optimization.json").write_text(
        json.dumps({"objective": "minimize_distance_to_safe_concise_suffix", "steps": args.steps, "losses": losses}, indent=2) + "\n",
        encoding="utf-8",
    )

    allowed_ids = suffix_ids[0].unique()
    allowed_embeddings = embedding_layer(allowed_ids).detach()
    local_indices = _nearest_allowed_tokens(state.detach()[0], allowed_embeddings)
    restored_suffix_ids = allowed_ids[local_indices]
    restored_suffix = tokenizer.decode(restored_suffix_ids.tolist(), skip_special_tokens=True)
    restored_prompt = initial_prompt + restored_suffix
    (args.output_dir / "04_materialized_prompt.json").write_text(
        json.dumps({
            "materialization": "nearest_neighbor_with_safe_suffix_token_constraint",
            "allowed_token_count": int(allowed_ids.numel()),
            "restored_suffix": restored_suffix,
            "restored_prompt": restored_prompt,
        }, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with torch.inference_mode():
        encoded = tokenizer(restored_prompt, return_tensors="pt").to(device)
        generated = model.generate(**encoded, max_new_tokens=96, do_sample=False)
    answer = tokenizer.decode(generated[0, encoded.input_ids.shape[1]:], skip_special_tokens=True)
    (args.output_dir / "05_generation.json").write_text(
        json.dumps({"max_new_tokens": 96, "response": answer}, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# Safe Embedding Demo\n\n"
        "A fixed-seed, allowlist-filtered Alpaca example. Only a concise-answer suffix is optimized; "
        "nearest-neighbor materialization is constrained to tokens from that suffix.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
