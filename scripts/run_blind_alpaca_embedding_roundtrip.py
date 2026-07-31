"""Run a semantically blind Alpaca output embedding roundtrip.

The runner intentionally performs no keyword filtering, language detection, or
content logging. It treats one randomly selected ``output`` string only as an
opaque prompt payload. Decoded prompt text is written to the result artifact,
never stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/alpaca_annotation_30k_zh_en/alpaca_en.json"
DEFAULT_MODEL = Path("/home/wh/models/qwen/Qwen2___5-7B-Instruct")
INITIAL_SOFT_PREFIX_TEXT = "Please respond to the following request: "
FROZEN_SEGMENT_1_TEXT = "\n[FIXED-1] System: You are a helpful assistant.\nUser: "
FROZEN_SEGMENT_2_TEXT = "\n[FIXED-2] Assistant: "
ASCII_COMMON_PUNCTUATION = frozenset(".,;:!?-'\"()[]{}")


def select_random_output(rows: list[object], seed: int) -> tuple[int, str]:
    """Select an ``output`` field by type only, without inspecting its text."""
    valid_indexes = [
        index
        for index, row in enumerate(rows)
        if isinstance(row, dict) and isinstance(row.get("output"), str)
    ]
    if not valid_indexes:
        raise ValueError("dataset contains no string output fields")
    index = random.Random(seed).choice(valid_indexes)
    row = rows[index]
    assert isinstance(row, dict)
    output = row["output"]
    assert isinstance(output, str)
    return index, output


def checkpoint_steps(steps: int, every: int) -> list[int]:
    """Return the required periodic checkpoints, including a final remainder."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if every < 1:
        raise ValueError("record interval must be positive")
    checkpoints = list(range(every, steps + 1, every))
    if steps not in checkpoints:
        checkpoints.append(steps)
    return checkpoints


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_embeddings_from_ids(embedding_layer: torch.nn.Module, source_ids: torch.Tensor) -> torch.Tensor:
    """Extract a detached target tensor that remains valid for later autograd."""
    with torch.no_grad():
        return embedding_layer(source_ids).detach().float()


def different_target_ids(source_ids: torch.Tensor, vocab_size: int, offset: int) -> torch.Tensor:
    """Map each source token to a deterministic, distinct vocabulary target."""
    if vocab_size < 2:
        raise ValueError("vocabulary must contain at least two tokens")
    normalized_offset = offset % vocab_size
    if normalized_offset == 0:
        raise ValueError("target offset must not be a multiple of the vocabulary size")
    return (source_ids + normalized_offset) % vocab_size


def combine_segment_text(soft_prefix: str, frozen_segment_1: str, editable_seed: str, frozen_segment_2: str) -> str:
    """Represent the materialized layout ``[z | frozen_1 | U | frozen_2]``."""
    return soft_prefix + frozen_segment_1 + editable_seed + frozen_segment_2


def initial_prompt_from_ids(
    tokenizer: Any,
    prefix_ids: torch.Tensor,
    frozen_segment_1: str,
    seed_ids: torch.Tensor,
    frozen_segment_2: str,
) -> str:
    """Materialize step zero directly from the original, unperturbed token IDs."""
    prefix = tokenizer.decode(prefix_ids.squeeze(0).tolist(), skip_special_tokens=True)
    seed = tokenizer.decode(seed_ids.squeeze(0).tolist(), skip_special_tokens=True)
    return combine_segment_text(prefix, frozen_segment_1, seed, frozen_segment_2)


def serialize_checkpoint_record(step: int, prompt: str) -> str:
    """Serialize only the checkpoint index and assembled prompt as UTF-8 text."""
    return json.dumps({"step": step, "prompt": prompt}, ensure_ascii=False)


def _is_readable_token_text(value: str) -> bool:
    return bool(value) and value.isprintable() and "\ufffd" not in value and any(char.isalnum() for char in value)


def _is_ascii_english_token_text(value: str) -> bool:
    return (
        bool(value)
        and value.isascii()
        and value.isprintable()
        and all(char.isalnum() or char.isspace() or char in ASCII_COMMON_PUNCTUATION for char in value)
        and any(char.isalnum() or char in ASCII_COMMON_PUNCTUATION for char in value)
    )


def ascii_english_vocabulary_ids(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Return token IDs that independently decode to ASCII English text or punctuation."""
    ids = [
        token_id
        for token_id in range(vocab_size)
        if _is_ascii_english_token_text(
            tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
    ]
    if len(ids) < 2:
        raise ValueError("vocabulary has fewer than two independently ASCII-readable tokens")
    return torch.tensor(ids, dtype=torch.long)


def readable_distinct_target_ids(
    source_ids: torch.Tensor, readable_ids: torch.Tensor, offset: int
) -> torch.Tensor:
    """Choose a deterministic readable target token different from each source token."""
    if readable_ids.ndim != 1 or readable_ids.numel() < 2:
        raise ValueError("readable token IDs must be a one-dimensional tensor with at least two entries")
    candidates = readable_ids.to(source_ids.device)
    positions = torch.remainder(source_ids + offset, candidates.numel())
    target_ids = candidates[positions]
    for _ in range(candidates.numel()):
        collisions = target_ids == source_ids
        if not torch.any(collisions):
            return target_ids
        positions = torch.where(collisions, (positions + 1) % candidates.numel(), positions)
        target_ids = candidates[positions]
    raise ValueError("readable vocabulary cannot provide a distinct target for every source token")


def progressive_target_mask(total_tokens: int, stage: int, stages: int) -> torch.Tensor:
    """Activate an additional contiguous token group for each optimization stage."""
    if total_tokens < 1:
        raise ValueError("total token count must be positive")
    if stages < 1 or not 0 <= stage <= stages:
        raise ValueError("stage must be within the configured stage count")
    active_count = (total_tokens * stage + stages - 1) // stages
    mask = torch.zeros(total_tokens, dtype=torch.bool)
    mask[:active_count] = True
    return mask


def frozen_chat_segments(tokenizer: Any, device: torch.device) -> tuple[str, torch.Tensor, str, torch.Tensor]:
    """Build two visible immutable segments around the editable user seed block."""
    frozen_1_ids = tokenizer(FROZEN_SEGMENT_1_TEXT, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    frozen_2_ids = tokenizer(FROZEN_SEGMENT_2_TEXT, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return FROZEN_SEGMENT_1_TEXT, frozen_1_ids, FROZEN_SEGMENT_2_TEXT, frozen_2_ids


def nearest_token_ids(
    embeddings: torch.Tensor,
    embedding_weight: torch.Tensor,
    chunk_size: int,
    allowed_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project embeddings to vocabulary IDs without materializing all scores."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (tokens, embedding_dim)")
    if chunk_size < 1:
        raise ValueError("vocabulary chunk size must be positive")

    normalized_queries = functional.normalize(embeddings.float(), dim=-1).to(embedding_weight.dtype)
    best_scores = torch.full(
        (normalized_queries.shape[0],), -torch.inf, device=embedding_weight.device, dtype=torch.float32
    )
    best_ids = torch.zeros(normalized_queries.shape[0], device=embedding_weight.device, dtype=torch.long)

    candidate_ids = (
        torch.arange(embedding_weight.shape[0], device=embedding_weight.device)
        if allowed_ids is None
        else allowed_ids.to(embedding_weight.device)
    )
    if candidate_ids.numel() == 0:
        raise ValueError("nearest-token projection requires at least one allowed token")
    for start in range(0, candidate_ids.numel(), chunk_size):
        chunk_ids = candidate_ids[start : start + chunk_size]
        chunk = embedding_weight[chunk_ids]
        normalized_chunk = functional.normalize(chunk.float(), dim=-1).to(embedding_weight.dtype)
        scores = (normalized_queries @ normalized_chunk.T).float()
        chunk_scores, chunk_positions = scores.max(dim=1)
        update = chunk_scores > best_scores
        best_scores = torch.where(update, chunk_scores, best_scores)
        best_ids = torch.where(update, chunk_ids[chunk_positions], best_ids)

    return best_ids


def _load_rows(dataset: Path) -> list[object]:
    loaded: Any = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("dataset root must be a JSON list")
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/blind_alpaca_embedding_roundtrip")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--record-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--initial-noise", type=float, default=0.02)
    parser.add_argument("--vocab-chunk-size", type=int, default=2048)
    parser.add_argument("--target-offset", type=int, default=104729)
    args = parser.parse_args()

    checkpoints = checkpoint_steps(args.steps, args.record_every)
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.initial_noise < 0:
        raise ValueError("initial noise must be non-negative")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_index, source_prompt = select_random_output(_load_rows(args.dataset), args.seed)
    source_hash = sha256_text(source_prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device

    source_ids = tokenizer(source_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if source_ids.shape[1] == 0:
        raise ValueError("selected output tokenizes to zero tokens")
    source_embeddings = source_embeddings_from_ids(embedding_layer, source_ids)
    readable_ids = ascii_english_vocabulary_ids(tokenizer, embedding_layer.weight.shape[0]).to(device)
    target_ids = readable_distinct_target_ids(source_ids, readable_ids, args.target_offset)
    target_embeddings = source_embeddings_from_ids(embedding_layer, target_ids)
    frozen_1, frozen_1_ids, frozen_2, frozen_2_ids = frozen_chat_segments(tokenizer, device)
    frozen_1_embeddings = source_embeddings_from_ids(embedding_layer, frozen_1_ids)
    frozen_2_embeddings = source_embeddings_from_ids(embedding_layer, frozen_2_ids)

    noise_generator = torch.Generator(device=device).manual_seed(args.seed)
    prefix_ids = tokenizer(
        INITIAL_SOFT_PREFIX_TEXT, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    if prefix_ids.shape[1] == 0:
        raise ValueError("initial soft prefix tokenizes to zero tokens")
    prefix_embeddings = source_embeddings_from_ids(embedding_layer, prefix_ids)
    prefix_target_ids = readable_distinct_target_ids(prefix_ids, readable_ids, args.target_offset)
    prefix_target_embeddings = source_embeddings_from_ids(embedding_layer, prefix_target_ids)
    prefix_state = (prefix_embeddings + args.initial_noise * torch.randn(
        prefix_embeddings.shape, generator=noise_generator, device=device, dtype=torch.float32
    )).detach().requires_grad_(True)
    seed_state = (source_embeddings + args.initial_noise * torch.randn(
        source_embeddings.shape, generator=noise_generator, device=device, dtype=torch.float32
    )).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([prefix_state, seed_state], lr=args.learning_rate)

    checkpoint_path = args.output_dir / "checkpoints.jsonl"
    seen_prompt_hashes: set[str] = set()

    def write_checkpoint(step: int, checkpoint_file: Any) -> None:
        if step == 0:
            prompt = initial_prompt_from_ids(tokenizer, prefix_ids, frozen_1, source_ids, frozen_2)
            prompt_hash = sha256_text(prompt)
            seen_prompt_hashes.add(prompt_hash)
            checkpoint_file.write(serialize_checkpoint_record(step, prompt) + "\n")
            checkpoint_file.flush()
            return
        with torch.inference_mode():
            restored_prefix_ids = nearest_token_ids(
                prefix_state.detach().squeeze(0), embedding_layer.weight.detach(), args.vocab_chunk_size, readable_ids
            )
            restored_seed_ids = nearest_token_ids(
                seed_state.detach().squeeze(0), embedding_layer.weight.detach(), args.vocab_chunk_size, readable_ids
            )
            restored_prefix = tokenizer.decode(restored_prefix_ids.tolist(), skip_special_tokens=True)
            restored_seed = tokenizer.decode(restored_seed_ids.tolist(), skip_special_tokens=True)
            prompt = combine_segment_text(restored_prefix, frozen_1, restored_seed, frozen_2)
            # Construct the exact continuous layout even though only z and U receive gradients.
            torch.cat([prefix_state.detach(), frozen_1_embeddings, seed_state.detach(), frozen_2_embeddings], dim=1)
        prompt_hash = sha256_text(prompt)
        if prompt_hash in seen_prompt_hashes:
            raise RuntimeError(f"checkpoint {step} did not produce a new materialized prompt")
        seen_prompt_hashes.add(prompt_hash)
        checkpoint_file.write(serialize_checkpoint_record(step, prompt) + "\n")
        checkpoint_file.flush()

    with checkpoint_path.open("w", encoding="utf-8") as checkpoint_file:
        write_checkpoint(0, checkpoint_file)
        for step in range(1, args.steps + 1):
            stage = next(index for index, checkpoint in enumerate(checkpoints, start=1) if step <= checkpoint)
            active_mask = progressive_target_mask(
                prefix_ids.shape[1] + source_ids.shape[1], stage, len(checkpoints)
            ).to(device)
            prefix_mask = active_mask[: prefix_ids.shape[1]].view(1, -1, 1)
            seed_mask = active_mask[prefix_ids.shape[1] :].view(1, -1, 1)
            stage_prefix_target = torch.where(prefix_mask, prefix_target_embeddings, prefix_embeddings)
            stage_seed_target = torch.where(seed_mask, target_embeddings, source_embeddings)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.mse_loss(prefix_state, stage_prefix_target) + functional.mse_loss(
                seed_state, stage_seed_target
            )
            loss.backward()
            optimizer.step()

            if step in checkpoints:
                write_checkpoint(step, checkpoint_file)

    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
