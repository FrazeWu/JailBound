"""Deterministic semantic helpers for constrained taxonomy assignment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from .io import canonical_hash


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class CalibrationParaphraseRejected(ValueError):
    """A held-out seed could not produce a constraint-preserving rewrite."""


class CalibrationSimilarityRejected(ValueError):
    """A held-out seed produced an unusable semantic similarity."""


class CalibrationPoolExhausted(ValueError):
    """A fixed calibration pool could not satisfy all source quotas."""

    def __init__(
        self,
        *,
        accepted_by_source: dict[str, int],
        rejected_paraphrase_by_source: dict[str, int],
        rejected_similarity_by_source: dict[str, int],
    ) -> None:
        super().__init__("calibration holdout pool could not satisfy every source quota")
        self.accepted_by_source = accepted_by_source
        self.rejected_paraphrase_by_source = rejected_paraphrase_by_source
        self.rejected_similarity_by_source = rejected_similarity_by_source


@dataclass(frozen=True)
class CalibrationPair:
    """Content-free semantic-calibration evidence for one held-out pair."""

    pair_id: str
    source: str
    positive: bool
    similarity: float


@dataclass(frozen=True)
class CalibrationCandidate:
    """One transient held-out intent used only while freezing calibration."""

    example_id: str
    source: str
    risk_category: str
    intent: str


@dataclass(frozen=True)
class CalibrationHoldout:
    """A held-out positive seed paired with a same-category negative intent."""

    candidate: CalibrationCandidate
    negative: CalibrationCandidate


def select_calibration_holdouts(
    candidates: tuple[CalibrationCandidate, ...],
    *,
    controlled_ids: frozenset[str],
    per_source: int,
    seed: int,
    allow_shortfall: bool = False,
) -> tuple[CalibrationHoldout, ...]:
    """Select deterministic, disjoint held-outs with matched negative partners."""

    if per_source < 1:
        raise ValueError("calibration holdout count must be positive")
    if not candidates:
        raise ValueError("calibration candidates cannot be empty")
    sources = {candidate.source for candidate in candidates}
    if len(sources) != 1:
        raise ValueError("calibration candidates must come from one source")
    ids = [candidate.example_id for candidate in candidates]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("calibration candidate ids must be non-empty and unique")
    available = tuple(candidate for candidate in candidates if candidate.example_id not in controlled_ids)
    by_category: dict[str, list[CalibrationCandidate]] = {}
    for candidate in available:
        by_category.setdefault(candidate.risk_category, []).append(candidate)
    eligible = tuple(
        candidate
        for category in sorted(by_category)
        if len(by_category[category]) >= 2
        for candidate in by_category[category]
    )
    ordered = sorted(
        eligible,
        key=lambda candidate: canonical_hash(
            {"seed": seed, "source": candidate.source, "example_id": candidate.example_id}
        ),
    )
    if len(ordered) < per_source and not allow_shortfall:
        raise ValueError("insufficient held-out same-category candidates")
    selected = ordered[:per_source]
    holdouts: list[CalibrationHoldout] = []
    for candidate in selected:
        partners = [
            partner
            for partner in by_category[candidate.risk_category]
            if partner.example_id != candidate.example_id
        ]
        negative = min(
            partners,
            key=lambda partner: canonical_hash(
                {
                    "seed": seed,
                    "candidate": candidate.example_id,
                    "negative": partner.example_id,
                }
            ),
        )
        holdouts.append(CalibrationHoldout(candidate=candidate, negative=negative))
    return tuple(holdouts)


def build_calibration_pairs(
    holdouts: tuple[CalibrationHoldout, ...],
    *,
    paraphrase: Callable[[str], str],
    category_for_text: Callable[[str], str],
    entities_preserved: Callable[[str, str], bool],
    similarity: Callable[[str, str], float],
    max_attempts: int,
) -> tuple[CalibrationPair, ...]:
    """Generate accepted positive pairs plus matched negatives without persisting text."""

    if not holdouts:
        raise ValueError("calibration holdouts cannot be empty")
    if max_attempts < 1:
        raise ValueError("calibration max_attempts must be positive")
    pairs: list[CalibrationPair] = []
    for holdout in holdouts:
        candidate, negative = holdout.candidate, holdout.negative
        if candidate.source != negative.source or candidate.risk_category != negative.risk_category:
            raise ValueError("calibration negatives must match source and risk category")
        rewritten: str | None = None
        for _ in range(max_attempts):
            proposal = paraphrase(candidate.intent)
            if (
                isinstance(proposal, str)
                and proposal.strip()
                and category_for_text(proposal) == candidate.risk_category
                and entities_preserved(candidate.intent, proposal)
            ):
                rewritten = proposal
                break
        if rewritten is None:
            raise CalibrationParaphraseRejected(
                f"could not produce an accepted calibration paraphrase for {candidate.example_id}"
            )
        positive_similarity = float(similarity(candidate.intent, rewritten))
        negative_similarity = float(similarity(candidate.intent, negative.intent))
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (positive_similarity, negative_similarity)
        ):
            raise CalibrationSimilarityRejected("calibration similarity must be a finite probability")
        identity = {"source": candidate.source, "example_id": candidate.example_id}
        pairs.extend(
            (
                CalibrationPair(
                    pair_id=f"calibration:{canonical_hash(identity | {'kind': 'positive'})[:20]}",
                    source=candidate.source,
                    positive=True,
                    similarity=positive_similarity,
                ),
                CalibrationPair(
                    pair_id=f"calibration:{canonical_hash(identity | {'kind': 'negative', 'negative_id': negative.example_id})[:20]}",
                    source=candidate.source,
                    positive=False,
                    similarity=negative_similarity,
                ),
            )
        )
    return tuple(pairs)


def build_calibration_pairs_from_pool(
    holdouts: tuple[CalibrationHoldout, ...],
    *,
    per_source: int,
    paraphrase: Callable[[str], str],
    category_for_text: Callable[[str], str],
    entities_preserved: Callable[[str, str], bool],
    similarity: Callable[[str, str], float],
    max_attempts: int,
) -> tuple[CalibrationPair, ...]:
    """Fill each source quota from a fixed pool, skipping rejected rewrites only."""

    if per_source < 1:
        raise ValueError("calibration holdout count must be positive")
    if not holdouts:
        raise ValueError("calibration holdout pool cannot be empty")
    sources = {holdout.candidate.source for holdout in holdouts}
    accepted_by_source = {source: 0 for source in sources}
    rejected_paraphrase_by_source = {source: 0 for source in sources}
    rejected_similarity_by_source = {source: 0 for source in sources}
    pairs: list[CalibrationPair] = []
    for holdout in holdouts:
        source = holdout.candidate.source
        if accepted_by_source[source] >= per_source:
            continue
        try:
            accepted_pairs = build_calibration_pairs(
                (holdout,),
                paraphrase=paraphrase,
                category_for_text=category_for_text,
                entities_preserved=entities_preserved,
                similarity=similarity,
                max_attempts=max_attempts,
            )
        except CalibrationParaphraseRejected:
            rejected_paraphrase_by_source[source] += 1
            continue
        except CalibrationSimilarityRejected:
            rejected_similarity_by_source[source] += 1
            continue
        pairs.extend(accepted_pairs)
        accepted_by_source[source] += 1
    if any(count != per_source for count in accepted_by_source.values()):
        raise CalibrationPoolExhausted(
            accepted_by_source=accepted_by_source,
            rejected_paraphrase_by_source=rejected_paraphrase_by_source,
            rejected_similarity_by_source=rejected_similarity_by_source,
        )
    return tuple(pairs)


def freeze_semantic_calibration(
    pairs: tuple[CalibrationPair, ...],
    *,
    target_recall: float,
    encoder_revision: str,
) -> dict[str, object]:
    """Freeze a semantic threshold without persisting held-out prompt text."""

    if not pairs:
        raise ValueError("semantic calibration requires pairs")
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    if not isinstance(encoder_revision, str) or not encoder_revision:
        raise ValueError("encoder_revision must be non-empty")
    pair_ids = [pair.pair_id for pair in pairs]
    if any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
        raise ValueError("calibration pair ids must be non-empty")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate calibration pair id")
    if any(not isinstance(pair.source, str) or not pair.source for pair in pairs):
        raise ValueError("calibration pair sources must be non-empty")
    if any(type(pair.positive) is not bool for pair in pairs):
        raise TypeError("calibration pair positive flags must be booleans")
    if any(not math.isfinite(pair.similarity) or not 0.0 <= pair.similarity <= 1.0 for pair in pairs):
        raise ValueError("calibration similarities must be finite probabilities")
    positives = [pair.similarity for pair in pairs if pair.positive]
    negatives = [pair.similarity for pair in pairs if not pair.positive]
    if not positives or not negatives:
        raise ValueError("semantic calibration requires both positive and negative pairs")
    retained = max(1, math.ceil(len(positives) * target_recall))
    threshold = sorted(positives, reverse=True)[retained - 1]
    return {
        "threshold": threshold,
        "target_positive_recall": target_recall,
        "positive_recall": sum(value >= threshold for value in positives) / len(positives),
        "negative_acceptance": sum(value >= threshold for value in negatives) / len(negatives),
        "encoder_revision": encoder_revision,
        "pair_count": len(pairs),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "pair_id_hash": canonical_hash(sorted(pair_ids)),
        "sources": sorted({pair.source for pair in pairs}),
    }


class SemanticEncoder:
    """Lazy BGE-M3 wrapper with normalized vectors and fixed input ordering."""

    def __init__(self, model_id: str = "BAAI/bge-m3", *, revision: str | None = None, batch_size: int = 32) -> None:
        self.model_id = model_id
        self.revision = revision
        self.resolved_revision = revision
        self.batch_size = batch_size
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, revision=self.revision)
            auto_model = getattr(self._model[0], "auto_model", None)
            config = getattr(auto_model, "config", None)
            self.resolved_revision = getattr(config, "_commit_hash", None) or self.revision
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = np.asarray(self._load().encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        return vectors


class QwenHiddenMeanEncoder:
    """Local compatibility encoder used when the approved BGE snapshot is unavailable."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        batch_size: int = 8,
        tokenizer: Any | None = None,
        model: Any | None = None,
        revision: str | None = None,
    ) -> None:
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be supplied together")
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.resolved_revision: str | None = revision
        self._tokenizer = tokenizer
        self._model = model
        if self._model is not None:
            self._model.eval()
            self.resolved_revision = self.resolved_revision or f"local:{Path(self.model_path).name}"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
        self._model.eval()
        self.resolved_revision = f"local:{Path(self.model_path).name}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        import torch
        self._load()
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self._tokenizer(
                texts[start:start + self.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._model_device())
            with torch.inference_mode():
                output = self._model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden_states = getattr(output, "hidden_states", None)
                if not hidden_states:
                    raise ValueError("semantic encoder did not return hidden states")
                hidden = hidden_states[-1]
            hidden = hidden.float()
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.extend(pooled.cpu().numpy())
        return np.asarray(vectors, dtype=np.float32)

    def _model_device(self) -> Any:
        if self._model is None:
            raise RuntimeError("semantic encoder model is unavailable")
        embedding_getter = getattr(self._model, "get_input_embeddings", None)
        if callable(embedding_getter):
            weight = getattr(embedding_getter(), "weight", None)
            device = getattr(weight, "device", None)
            if device is not None:
                return device
        device = getattr(self._model, "device", None)
        if device is None:
            raise ValueError("semantic encoder model has no input device")
        return device


def load_taxonomy_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("taxonomy mapping must contain a YAML object")
    return payload


def choose_canonical_label(text: str, candidates: list[str], label_embeddings: dict[str, np.ndarray], encoder: Encoder) -> tuple[str, dict[str, float]]:
    if not candidates:
        raise ValueError("canonical label candidates cannot be empty")
    missing = [label for label in candidates if label not in label_embeddings]
    if missing:
        raise ValueError(f"missing label embeddings: {', '.join(missing)}")
    vector = encoder.encode([text])[0]
    scores = {label: float(vector @ label_embeddings[label]) for label in candidates}
    winner = sorted(scores, key=lambda label: (-scores[label], label))[0]
    return winner, scores


@dataclass(frozen=True)
class MappingDecision:
    """One auditable canonical assignment for a raw source record."""

    risk_category: str
    attack_type: str
    threat_domain: str
    routes: dict[str, str]
    candidate_scores: dict[str, dict[str, float]]

    @property
    def audit_entry(self) -> dict[str, object]:
        """Structured metadata for the manifest-build mapping report."""
        return {
            "risk_category": self.risk_category,
            "attack_type": self.attack_type,
            "threat_domain": self.threat_domain,
            "routes": self.routes,
            "candidate_scores": self.candidate_scores,
        }

    @property
    def preprocessing(self) -> tuple[str, ...]:
        entries = []
        for field in ("risk_category", "attack_type", "threat_domain"):
            entries.append(f"mapping_route:{field}:{self.routes[field]}")
            if self.candidate_scores[field]:
                entries.append(
                    f"mapping_scores:{field}:"
                    f"{json.dumps(self.candidate_scores[field], sort_keys=True, separators=(',', ':'))}"
                )
        return tuple(entries)


def _mapped_label(
    *,
    text: str,
    exact: str | None,
    candidates: list[str] | None,
    route_if_exact: str,
    label_embeddings: dict[str, np.ndarray],
    encoder: Encoder,
) -> tuple[str, str, dict[str, float]]:
    if exact is not None:
        return exact, route_if_exact, {}
    if candidates is None:
        raise ValueError("mapping has neither an exact label nor constrained candidates")
    label, scores = choose_canonical_label(text, candidates, label_embeddings, encoder)
    return label, "constrained_bge", scores


def map_raw_example(
    raw: Any,
    mapping: dict[str, Any],
    label_embeddings: dict[str, np.ndarray],
    encoder: Encoder,
) -> MappingDecision:
    """Map one adapter record and preserve the route and all BGE scores."""
    source = raw.source
    source_risk = raw.source_risk_label
    exact_risks = mapping.get("exact_risk_category", {}).get(source, {})
    broad_risks = mapping.get("broad_risk_candidates", {})
    if source == "jailbound":
        exact_risk = mapping["jailbound_risk_category"].get(source_risk)
        risk_candidates = None
    elif source == "s_eval":
        exact_risk = mapping["s_eval_risk_category"].get(source_risk)
        risk_candidates = None
    elif source == "advbench":
        exact_risk = None
        risk_candidates = broad_risks["advbench.unlabeled"]
    else:
        exact_risk = exact_risks.get(source_risk)
        risk_candidates = broad_risks.get(f"{source}.{source_risk}")
    risk, risk_route, risk_scores = _mapped_label(
        text=raw.intent,
        exact=exact_risk,
        candidates=risk_candidates,
        route_if_exact="exact",
        label_embeddings=label_embeddings,
        encoder=encoder,
    )

    if raw.source_attack_label == "direct_request":
        attack, attack_route, attack_scores = "direct_request", "direct_request_control", {}
    elif source == "jailbound":
        attack, attack_route, attack_scores = _mapped_label(
            text=raw.attack_text,
            exact=mapping["jailbound_attack_type"].get(raw.source_attack_label),
            candidates=None,
            route_if_exact="native_attack_map",
            label_embeddings=label_embeddings,
            encoder=encoder,
        )
    elif source == "s_eval":
        attack, attack_route, attack_scores = _mapped_label(
            text=raw.attack_text,
            exact=mapping["s_eval_attack_type"].get(raw.source_attack_label),
            candidates=None,
            route_if_exact="native_attack_map",
            label_embeddings=label_embeddings,
            encoder=encoder,
        )
    else:
        raise ValueError(f"unsupported source attack label for {source}")

    native_domains = {
        value["source_label"]: key for key, value in mapping["threat_domains"].items()
    }
    domain, domain_route, domain_scores = _mapped_label(
        text=raw.attack_text,
        exact=native_domains.get(raw.source_domain_label),
        candidates=None if raw.source_domain_label in native_domains else list(mapping["threat_domains"]),
        route_if_exact="exact",
        label_embeddings=label_embeddings,
        encoder=encoder,
    )
    return MappingDecision(
        risk_category=risk,
        attack_type=attack,
        threat_domain=domain,
        routes={"risk_category": risk_route, "attack_type": attack_route, "threat_domain": domain_route},
        candidate_scores={"risk_category": risk_scores, "attack_type": attack_scores, "threat_domain": domain_scores},
    )
