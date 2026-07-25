"""
QuoTe Model Loader Module

Loads and configures the target white-box model (Qwen3-4B-Instruct-2507) for
compliance boundary detection. All model weights are frozen; no fine-tuning occurs.

Design constraints:
- model.eval() enforced — no dropout or batch-norm effects
- model.requires_grad_(False) — only embedding tensors passed to this module
  have gradients enabled by the caller
- Single model only: Qwen/Qwen3-4B-Instruct-2507
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# Qwen3 special token strings — used when building perturbation masks
QWEN3_SPECIAL_TOKENS = {
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
}

# Role strings that appear immediately after <|im_start|>
QWEN3_ROLE_STRINGS = {"system", "user", "assistant"}


@dataclass
class LoadedModel:
    """
    Container for a frozen, evaluation-mode model and its tokenizer.

    Attributes:
        model: The frozen causal LM (Qwen3-4B).
        tokenizer: Corresponding tokenizer.
        device: Torch device the model is on.
        embed_layer: Shortcut to model.get_input_embeddings().
        vocab_size: Actual vocabulary size from the embedding weight matrix.
        special_token_ids: Set of token IDs that must NOT be perturbed.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    embed_layer: torch.nn.Embedding
    vocab_size: int
    special_token_ids: frozenset[int]


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(
            f"Unsupported torch_dtype '{dtype_str}'. Choose from {list(mapping)}"
        )
    return mapping[dtype_str]


def _collect_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """
    Collect all token IDs that represent structural/special tokens which must
    not be perturbed during embedding-space search.

    Includes:
    - BOS, EOS, PAD, UNK tokens
    - <|im_start|> and <|im_end|> (Qwen3 chat template markers)
    - Explicit all_special_ids from the tokenizer
    """
    ids: set[int] = set()

    # Standard special tokens
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(int(tid))

    # Tokenizer's declared special ids
    if hasattr(tokenizer, "all_special_ids"):
        ids.update(int(i) for i in tokenizer.all_special_ids)

    # Qwen3 chat-template special strings
    for tok_str in QWEN3_SPECIAL_TOKENS:
        try:
            encoded = tokenizer.encode(tok_str, add_special_tokens=False)
            ids.update(int(i) for i in encoded)
        except Exception:
            pass

    # Role strings ("system", "user", "assistant") as single tokens
    for role in QWEN3_ROLE_STRINGS:
        try:
            encoded = tokenizer.encode(role, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
        except Exception:
            pass

    return frozenset(ids)


def load_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    load_in_8bit: bool = False,
    max_memory: dict | None = None,
) -> LoadedModel:
    """
    Load and freeze the target model for compliance boundary detection.

    The model is placed in eval mode with all weight gradients disabled.
    Only embedding tensors explicitly created by the caller will carry gradients.

    Args:
        model_path: HuggingFace model ID or local directory path.
        device: Target device string (e.g. "cuda", "cuda:0", "cpu").
        torch_dtype: Model weight dtype — "bfloat16" (default), "float16", or "float32".
        trust_remote_code: Passed to from_pretrained(); required for Qwen3.
        load_in_8bit: If True, load with bitsandbytes int8 quantization (~70B models).

    Returns:
        LoadedModel with frozen model, tokenizer, and derived metadata.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    use_device_map_auto = device in ("auto", "balanced", "balanced_low_0", "sequential")

    if not use_device_map_auto:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Use --device cpu or ensure GPU is available."
            )
    else:
        resolved_device = None  # will be set after model loads

    dtype = _resolve_dtype(torch_dtype)
    logger.info("Loading tokenizer from '%s'", model_path)
    # local_files_only=True prevents HF Hub from interpreting paths with dots
    # (e.g. "qwen3-72b") as namespaced repos.
    is_local_path = os.path.isdir(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
        local_files_only=is_local_path,
    )
    if tokenizer.pad_token_id is None:
        # Qwen3 uses eos as pad by convention
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug(
            "pad_token_id was None; set to eos_token_id=%d", tokenizer.eos_token_id
        )

    logger.info(
        "Loading model from '%s' with dtype=%s on device=%s%s", model_path, dtype, device,
        " [int8]" if load_in_8bit else "",
    )
    from_pretrained_kwargs: dict = dict(
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=device,
        local_files_only=is_local_path,
    )
    if max_memory is not None:
        from_pretrained_kwargs["max_memory"] = max_memory
    if load_in_8bit:
        from_pretrained_kwargs["load_in_8bit"] = True
        # int8 requires device_map; device_map="auto" is already set above
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_path,
        **from_pretrained_kwargs,
    )

    if use_device_map_auto:
        # For multi-GPU models, use the device of the first parameter
        resolved_device = next(model.parameters()).device

    # -------------------------------------------------------------------------
    # Freeze ALL model parameters — QuoTe is a white-box evaluation method,
    # not a training method.  Gradients flow only through temporary embedding
    # tensors created by the perturbation module.
    # -------------------------------------------------------------------------
    model.eval()
    model.requires_grad_(False)
    logger.info("Model frozen: eval() + requires_grad_(False)")

    embed_layer: torch.nn.Embedding = model.get_input_embeddings()
    # Actual vocab size may exceed tokenizer.vocab_size for some models
    vocab_size: int = embed_layer.weight.shape[0]

    special_ids = _collect_special_token_ids(tokenizer)
    logger.info(
        "Collected %d special/structural token IDs that will be excluded from perturbation",
        len(special_ids),
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        embed_layer=embed_layer,
        vocab_size=vocab_size,
        special_token_ids=special_ids,
    )


def verify_model_frozen(loaded: LoadedModel) -> None:
    """
    Assert that all model parameters have requires_grad=False.

    Args:
        loaded: LoadedModel to verify.

    Raises:
        AssertionError: If any parameter has requires_grad=True.
    """
    trainable = [name for name, p in loaded.model.named_parameters() if p.requires_grad]
    assert not trainable, (
        f"Model has {len(trainable)} trainable parameter(s) — weights must be frozen. "
        f"First offender: {trainable[0]}"
    )
    logger.debug("Model weight freeze verified: 0 trainable parameters.")


def get_embed_dim(loaded: LoadedModel) -> int:
    """Return the embedding dimension d of the model."""
    return int(loaded.embed_layer.weight.shape[1])
