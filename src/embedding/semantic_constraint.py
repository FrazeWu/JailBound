"""
Semantic Constraint Module

Computes L2-squared semantic distance between original and mutated seeds to
enforce attack-intent preservation during seed rewriting.

    D_L2(s', s) = || g(s') - g(s) ||_2^2

where g(s) = Pool(E(Tok(s))) is the mean-pooled token embedding (no forward
pass — uses only the embedding lookup table).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from materialization.model_loader import LoadedModel

logger = logging.getLogger(__name__)


class SemanticEncoder:
    """Sentence encoder using the surrogate model's token embedding layer.

    g(s) = Pool(E(Tok(s))) — mean-pooled token embeddings, unnormalised.
    Much cheaper than a full forward pass (just an embedding lookup).
    """

    def __init__(self, loaded: LoadedModel) -> None:
        self.embed_layer = loaded.embed_layer
        self.tokenizer = loaded.tokenizer
        self.device = loaded.device

    @torch.no_grad()
    def encode(self, text: str, max_length: int = 256) -> torch.Tensor:
        """Encode a string into a mean-pooled token embedding vector.

        Returns:
            (d,) float32 tensor on CPU (unnormalised).
        """
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        embeds = self.embed_layer(input_ids)  # (1, T, d)
        mask = attention_mask.unsqueeze(-1).float()  # (1, T, 1)
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0).float().cpu()

    @torch.no_grad()
    def encode_batch(self, texts: list[str], max_length: int = 256) -> torch.Tensor:
        """Encode multiple strings.

        Returns:
            (N, d) float32 tensor on CPU (unnormalised).
        """
        if not texts:
            return torch.empty(0)
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            add_special_tokens=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        embeds = self.embed_layer(input_ids)  # (N, T, d)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.float().cpu()


def compute_semantic_distance(
    encoder: SemanticEncoder,
    text_a: str,
    text_b: str,
) -> float:
    """L2-squared distance between mean-pooled token embeddings.

    D_L2(a, b) = || g(a) - g(b) ||_2^2

    Returns a non-negative float. 0 = identical embeddings.
    """
    va = encoder.encode(text_a)
    vb = encoder.encode(text_b)
    return (va - vb).pow(2).sum().item()


def filter_candidates(
    encoder: SemanticEncoder,
    original_seed: str,
    candidates: list[str],
    epsilon_sem: float,
) -> list[tuple[str, float]]:
    """Filter seed-rewrite candidates by L2-squared distance threshold.

    Args:
        encoder: Semantic encoder instance.
        original_seed: The reference seed text.
        candidates: List of rewritten candidate strings.
        epsilon_sem: Maximum allowed L2-squared distance.

    Returns:
        List of (candidate_text, distance) tuples that pass the filter,
        sorted by distance ascending.
    """
    if not candidates:
        return []

    orig_vec = encoder.encode(original_seed)  # (d,)
    cand_vecs = encoder.encode_batch(candidates)  # (N, d)

    # L2-squared distances
    diff = cand_vecs - orig_vec.unsqueeze(0)  # (N, d)
    distances = diff.pow(2).sum(dim=1).tolist()  # (N,)

    passed: list[tuple[str, float]] = []
    for cand, dist in zip(candidates, distances):
        if dist <= epsilon_sem:
            passed.append((cand, dist))

    passed.sort(key=lambda x: x[1])
    logger.debug(
        "Semantic filter: %d / %d candidates passed (ε_sem=%.3f)",
        len(passed), len(candidates), epsilon_sem,
    )
    return passed
