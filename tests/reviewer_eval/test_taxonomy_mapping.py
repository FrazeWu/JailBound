from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.reviewer_eval.datasets import RawExample
from benchmark.reviewer_eval.semantic import choose_canonical_label, load_taxonomy_mapping, map_raw_example


ROOT = Path(__file__).resolve().parents[2]


class FakeEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        assert texts == ["sample"]
        return np.array([[0.0, 1.0]], dtype=np.float32)


def test_mapping_snapshot_contains_fixed_vocabularies() -> None:
    mapping = load_taxonomy_mapping(ROOT / "configs/benchmark/reviewer_taxonomy_map.yaml")
    assert len(mapping["risk_categories"]) == 12
    assert len(mapping["threat_domains"]) == 10
    assert len(mapping["attack_types"]) == 10
    assert mapping["s_eval_attack_type"]["positive_induction"] == "persuasion_deception"
    assert mapping["s_eval_attack_type"]["goal_hijacking"] == "prefix_injection"


def test_constrained_bge_selection_is_deterministic_and_exposes_scores() -> None:
    label, scores = choose_canonical_label("sample", ["beta", "alpha"], {"alpha": np.array([0.0, 1.0]), "beta": np.array([0.0, 1.0])}, FakeEncoder())
    assert label == "alpha"
    assert scores == {"beta": 1.0, "alpha": 1.0}


def test_mapping_records_routes_and_only_uses_bge_within_allowed_candidates() -> None:
    mapping = load_taxonomy_mapping(ROOT / "configs/benchmark/reviewer_taxonomy_map.yaml")
    raw = RawExample("advbench", 0, "sample", "sample", None, None, "direct_request", None, "en", ())
    embeddings = {label: np.array([0.0, 1.0]) for label in mapping["risk_categories"] | mapping["threat_domains"]}
    decision = map_raw_example(raw, mapping, embeddings, FakeEncoder())
    assert decision.attack_type == "direct_request"
    assert decision.routes == {"risk_category": "constrained_bge", "attack_type": "direct_request_control", "threat_domain": "constrained_bge"}
    assert set(decision.candidate_scores["risk_category"]) == set(mapping["broad_risk_candidates"]["advbench.unlabeled"])
    assert any(entry.startswith("mapping_scores:risk_category:") for entry in decision.preprocessing)
    assert decision.audit_entry == {
        "risk_category": decision.risk_category,
        "attack_type": decision.attack_type,
        "threat_domain": decision.threat_domain,
        "routes": decision.routes,
        "candidate_scores": decision.candidate_scores,
    }
